import os
import cv2
import tqdm
import multiprocessing
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd
import albumentations as albu
import segmentation_models_pytorch as smp

from src.radam import RAdam
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torchvision.transforms import ToTensor

import catalyst.utils as utils
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback, MixupCallback

NFOLDS = 5
RANDOM_STATE = 42
script_name = os.path.basename(__file__).split('.')[0]
skf = StratifiedKFold(n_splits=NFOLDS, random_state=RANDOM_STATE)
script_name = os.path.basename(__file__).split('.')[0]
MODEL_NAME = "{0}__folds{1}".format(script_name, NFOLDS)

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
        albu.Resize(350, 525),
        albu.PadIfNeeded(352, 544, border_mode=0)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(350, 525),
        albu.PadIfNeeded(352, 544, border_mode=0)
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
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label, shape)
            masks[:, :, idx] = mask

    return masks


class CloudDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None,
                 transforms=albu.Compose([albu.HorizontalFlip(), ToTensor()]),
                 preprocessing=None):
        self.df = df
        self.datatype = datatype
        self.img_ids = img_ids
        self.mask_dir = f"data/train_images_annots_350x525"
        self.transforms = transforms
        self.preprocessing = preprocessing
        # Optionally preload images for faster training
        if datatype != 'test':
            self.data_folder = f"data/train_images"
            self.samples = self.load_samples()
            self.raw_masks = self.load_masks()
        else:
            self.data_folder = f"data/test_images"

    def load_masks(self):
        masks = []
        for image_name in tqdm.tqdm(self.img_ids):
            encoded_masks = self.df.loc[self.df['im_id'] == image_name, 'EncodedPixels']
            masks.append(encoded_masks)

        return masks

    def load_samples(self):
        samples = []
        for img_id in tqdm.tqdm(self.img_ids):
            img = get_img(img_id, self.data_folder)
            samples.append(img)

        return samples

    def __getitem__(self, idx):

        if self.datatype != 'test':
            img = self.samples[idx]
            encoded_masks = self.raw_masks[idx]
            mask = make_mask(encoded_masks)
        else:
            img_id = self.img_ids[idx]
            img = get_img(img_id, self.data_folder)
            mask = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.float32)

        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)

    def _clear(self):
        del self.samples[:]


def train_model(args):
    # HACK: It's a pain to pass multiple arguments.
    # However, we can pass a single tuple and break it apart
    train_ids, valid_ids, logdir = args

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')

    num_workers = 0
    bs = 16
    train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, transforms=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, transforms=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    num_epochs = 25

    # model, criterion, optimizer
    optimizer = RAdam([
        {'params': model.decoder.parameters(), 'lr': 1e-2},
        {'params': model.encoder.parameters(), 'lr': 1e-3},
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2, threshold=0.001)
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)

    runner = SupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[DiceCallback(), MixupCallback(alpha=0.4), EarlyStoppingCallback(patience=5, min_delta=0.001)],
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )

    return True


def generate_valid_preds(args):

    train_ids, valid_ids, logdir = args
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
    valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, transforms=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
    )

    runner = SupervisedRunner()
    # Generate validation predictions
    loaders = {"infer": valid_loader}
    runner.infer(
        model=model,
        loaders=loaders,
        callbacks=[
            CheckpointCallback(
                resume=f"{logdir}/checkpoints/best.pth"),
            InferCallback()
        ],
    )

    valid_preds = np.load('data/valid_preds.npy')

    for im_id, preds in zip(valid_ids, runner.callbacks[0].predictions["logits"]):

        preds = preds[:, :350, :525]
        indexes = train.index[train['im_id'] == im_id]
        valid_preds[indexes[0]] = preds[0]  # fish
        valid_preds[indexes[1]] = preds[1]  # flower
        valid_preds[indexes[2]] = preds[2]  # gravel
        valid_preds[indexes[3]] = preds[3]  # sugar

    np.save('data/valid_preds.npy', valid_preds)

    return True


def get_valid_score(args):

    # Load predictions
    valid_preds = np.load('data/valid_preds.npy')

    # Load labels/masks
    all_ids = train[::4]['im_id']
    valid_masks = []
    for img_id in all_ids:
        encoded_masks = train.loc[train['im_id'] == img_id, 'EncodedPixels']
        mask = make_mask(encoded_masks)
        mask = cv2.resize(mask, (525, 350))  #height and width are backward in cv2...
        mask = mask.transpose(2, 0, 1)
        valid_masks.append(mask[0])
        valid_masks.append(mask[1])
        valid_masks.append(mask[2])
        valid_masks.append(mask[3])

    class_params = {}
    dice_scores = []

    for class_id in range(4):
        print(class_id)
        attempts = []
        for t in range(30, 100, 5):
            t /= 100
            for ms in [0, 1000, 5000, 10000]:
                masks = []
                for i in range(class_id, len(valid_preds), 4):
                    probability = valid_preds[i]
                    predict, num_predict = post_process(sigmoid(probability), t, ms)
                    masks.append(predict)

                d = []

                for i, j in zip(masks, valid_masks[class_id::4]):
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        d.append(dice(i, j))

                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

        attempts_df = attempts_df.sort_values('dice', ascending=False)
        print(attempts_df.head())
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]
        best_dice = attempts_df['dice'].values[0]
        dice_scores.append(best_dice)

        class_params[class_id] = (best_threshold, best_size)

    return (np.mean(dice_scores), class_params)


def generate_test_preds(args):

    valid_dice, class_params, = args

    test_preds = np.zeros((len(sub), 350, 525), dtype=np.float32)

    for i in range(NFOLDS):
        logdir = "./logs/segmentation_" + str(i)

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
        dummy_dataset = CloudDataset(df=sub, datatype='test', img_ids=test_ids[:1],  transforms=get_validation_augmentation(),
                                    preprocessing=get_preprocessing(preprocessing_fn))
        dummy_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=False, num_workers=0)

        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=4,
            activation=ACTIVATION,
        )
        runner = SupervisedRunner(model)

        # HACK: We are loading a few examples from our dummy loader so catalyst will properly load the weights
        # from our checkpoint
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
                preds = preds[:, :350, :525]
                idx = batch_index * 4
                test_preds[idx + 0] += sigmoid(preds[0]) / NFOLDS  # fish
                test_preds[idx + 1] += sigmoid(preds[1]) / NFOLDS  # flower
                test_preds[idx + 2] += sigmoid(preds[2]) / NFOLDS  # gravel
                test_preds[idx + 3] += sigmoid(preds[3]) / NFOLDS  # sugar

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
    sub.to_csv('submission_{}.csv'.format(valid_dice), columns=['Image_Label', 'EncodedPixels'], index=False)
    print("Saved.")


train = pd.read_csv(f'data/train.csv')
sub = pd.read_csv(f'data/sample_submission.csv')

train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])

id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(
    lambda x: x.split('_')[0]).value_counts().reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'satellite-resnet-50'
DEVICE = 'cuda'
ACTIVATION = None

i = 0

# Persist valid preds to disk
valid_preds = np.zeros((len(train), 350, 525), dtype=np.float32)
np.save('data/valid_preds.npy', valid_preds)
del valid_preds

for train_index, valid_index in skf.split(id_mask_count['img_id'].values, id_mask_count['count']):
    logdir = "./logs/segmentation_" + str(i)

    train_ids = train.iloc[train_index * 4]['im_id']
    valid_ids = train.iloc[valid_index * 4]['im_id']

    # Train model on fold
    with multiprocessing.Pool(1) as p:
        result = p.map(train_model, [(train_ids, valid_ids, logdir)])[0]
        print("Trained for fold ", str(i))

    # Get validation predictions
    with multiprocessing.Pool(1) as p:
        result = p.map(generate_valid_preds, [(train_ids, valid_ids, logdir)])[0]
        print("Generated validation preds for ", str(i))

    i = i + 1

# Get valid preds and optimize threshold and min_size
class_params = {}
with multiprocessing.Pool(1) as p:
    result = p.map(get_valid_score, [1])[0]
    valid_dice, class_params = result
    print("Valid Dice", valid_dice)
    print("Got back", class_params)

valid_dice = 0.5
class_params = {0: (0.75, 10000), 1: (0.7, 10000), 2: (0.7, 10000), 3: (0.6, 10000)}
# Generate test predictions
with multiprocessing.Pool(1) as p:
    result = p.map(generate_test_preds, [(valid_dice, class_params)])
    print(result)

print("done")
