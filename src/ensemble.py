import os
import cv2
import tqdm
import multiprocessing
from multiprocessing import Process

import numpy as np
import pandas as pd
import albumentations as albu
import segmentation_models_pytorch as smp

from src.radam import RAdam
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torchvision.transforms import ToTensor

import catalyst.utils as utils
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback

script_name = os.path.basename(__file__).split('.')[0]
MODEL_NAME = script_name

if os.getcwd().endswith('src'):
    # We want to be working in the root directory
    os.chdir('../')

print("Working dir", os.getcwd())
print("Model: {}".format(MODEL_NAME))


def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    """
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def mask2rle(img):
    """
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=(0.1, 0.1), rotate_limit=45, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.RandomContrast(limit=0.3, p=0.5),
        #albu.Resize(448, 672),
        #albu.PadIfNeeded(352, 544, border_mode=0)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        #albu.Resize(448, 672),
        #albu.PadIfNeeded(352, 544, border_mode=0),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_function):
    """Construct preprocessing transform

    Args:
        preprocessing_function (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_function),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


def get_img(name, image_dir):
    """
    Return image based on image name and folder.
    """
    image_path = os.path.join(image_dir, name)

    # read the data from the file
    with open(image_path, 'rb') as infile:
        buf = infile.read()

    # use numpy to construct an array from the bytes
    x = np.frombuffer(buf, dtype='uint8')

    # decode the array into an image
    img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)

    del infile
    del buf
    del x

    return img


def make_mask(encoded_masks, shape: tuple = (1400, 2100)):
    """
    Create mask based on df, image name and shape.
    """
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks


class CloudDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None,
                 transforms=albu.Compose([albu.HorizontalFlip(), ToTensor()]),
                 preprocessing=None):
        self.df = df
        self.datatype = datatype
        if datatype != 'test':
            self.data_folder = f"data/train_images"
        else:
            self.data_folder = f"data/test_images"
        self.img_ids = img_ids
        self.mask_dir = f"data/train_images_annots_350x525"
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.samples = self.load_samples()
        self.raw_masks = self.load_masks()

    def load_masks(self):
        masks = []
        for image_name in tqdm.tqdm(self.img_ids):
            encoded_masks = self.df.loc[self.df['im_id'] == image_name, 'EncodedPixels']
            mask = make_mask(encoded_masks)
            small_mask = cv2.resize(mask, (672, 448))
            del mask
            masks.append(small_mask)

        return masks

    def load_samples(self):
        samples = []
        for image_name in tqdm.tqdm(self.img_ids):
            img = get_img(image_name, self.data_folder)
            img = cv2.resize(img, (672, 448))
            samples.append(img)

        return samples

    def __getitem__(self, idx):

        img = self.samples[idx]

        if self.datatype != 'test':
            mask = self.raw_masks[idx]
        else:
            mask = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.float32)

        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']

        # We're resizing the mask because we're predicting with one less decoder
        # mask = mask.transpose((1, 2, 0))
        # mask = cv2.resize(mask, (544//2, 352//2))  # height and width are backward in cv2...
        # mask = mask.transpose((2, 0, 1))
        return img, mask

    def __len__(self):
        return len(self.img_ids)

    def _clear(self):
        del self.samples[:]


sub = pd.read_csv(f'data/sample_submission.csv')
sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

ACTIVATION = None
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

# TODO: Create list/dictionary of logdirs with models we'd like to assemble
# ENCODER = 'efficientnet-b2'
# logdir = "./logs/segmentation"

def generate_test_preds(ensemble_info):

    test_preds = np.zeros((len(sub), 350, 525), dtype=np.float32)
    num_models = len(ensemble_info)

    for model_info in ensemble_info:

        class_params = model_info['class_params']
        encoder = model_info['encoder']
        model_type = model_info['model_type']
        logdir = model_info['logdir']

        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS)

        model = None
        if model_type == 'unet':
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=ENCODER_WEIGHTS,
                classes=4,
                activation=ACTIVATION,
            )
        elif model_type == 'fpn':
            model = smp.FPN(
                encoder_name=encoder,
                encoder_weights=ENCODER_WEIGHTS,
                classes=4,
                activation=ACTIVATION,
            )
        else:
            raise NotImplementedError("We only support FPN and UNet")

        runner = SupervisedRunner(model)

        # HACK: We are loading a few examples from our dummy loader so catalyst will properly load the weights
        # from our checkpoint
        dummy_dataset = CloudDataset(df=sub, datatype='test', img_ids=test_ids[:1],  transforms=get_validation_augmentation(),
                                     preprocessing=get_preprocessing(preprocessing_fn))
        dummy_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=False, num_workers=0)
        loaders = {"test": dummy_loader}
        runner.infer(
            model=model,
            loaders=loaders,
            callbacks=[
                CheckpointCallback(
                    resume=f"{logdir}/checkpoints/best.pth"),
                InferCallback()
            ],
        )

        # Now we do real inference on the full dataset
        test_dataset = CloudDataset(df=sub, datatype='test', img_ids=test_ids,  transforms=get_validation_augmentation(),
                                    preprocessing=get_preprocessing(preprocessing_fn))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        image_id = 0
        for batch_index, test_batch in enumerate(tqdm.tqdm(test_loader)):
            runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits'].cpu().detach().numpy()
            for preds in runner_out:

                preds = preds.transpose((1, 2, 0))
                preds = cv2.resize(preds, (525, 350))  # height and width are backward in cv2...
                preds = preds.transpose((2, 0, 1))

                idx = batch_index * 4
                test_preds[idx + 0] += sigmoid(preds[0]) / num_models   # fish
                test_preds[idx + 1] += sigmoid(preds[1]) / num_models   # flower
                test_preds[idx + 2] += sigmoid(preds[2]) / num_models   # gravel
                test_preds[idx + 3] += sigmoid(preds[3]) / num_models   # sugar

    # Convert ensembled predictions to RLE predictions
    encoded_pixels = []
    for image_id, preds in enumerate(test_preds):

        predict, num_predict = post_process(preds, class_params[image_id % 4][0], class_params[image_id % 4][1])
        if num_predict == 0:
            encoded_pixels.append('')
        else:
            r = mask2rle(predict)
            encoded_pixels.append(r)

    print("Saving submission...")
    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv('ensembled_submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
    print("Saved.")


ensemble_info = 0  # TODO

# Generate test predictions
with multiprocessing.Pool(1) as p:
    result = p.map(generate_test_preds, [ensemble_info])


