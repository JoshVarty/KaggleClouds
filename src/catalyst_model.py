import os
import cv2
import tqdm
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
        albu.RandomCrop(512, 512)
    ]

    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(1408, 2112)  # Validate/test on full size images
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_function):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
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
            masks.append(encoded_masks)

        return masks

    def load_samples(self):
        samples = []
        for image_name in tqdm.tqdm(self.img_ids):
            img = get_img(image_name, self.data_folder)
            samples.append(img)

        return samples

    def __getitem__(self, idx):

        img = self.samples[idx]
        image_name = self.img_ids[idx]

        if self.datatype != 'test':
            encoded_masks = self.raw_masks[idx]
            mask = make_mask(encoded_masks)
        else:
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


def train():
    train = pd.read_csv(f'data/train.csv')
    sub = pd.read_csv(f'data/sample_submission.csv')

    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])

    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(
        lambda x: x.split('_')[0]).value_counts().reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)
    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = 'cuda'

    ACTIVATION = None
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    num_workers = 0
    bs = 8
    train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, transforms=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, transforms=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    num_epochs = 1
    logdir = "./logs/segmentation"

    # model, criterion, optimizer
    optimizer = RAdam([
        {'params': model.decoder.parameters(), 'lr': 1e-2},
        {'params': model.encoder.parameters(), 'lr': 1e-3},
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    runner = SupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )

    return True


# multiprocessing_pool = multiprocessing.Pool(1)
# result = multiprocessing_pool.map(train, [])
#
# multiprocessing_pool.terminate()
# multiprocessing_pool.join()

p = Process(target=train)
p.start()
p.join()

myVariable = 5

xx = np.random.randn(2000,3,350,525)

myVariable = 6