import cv2

import torch
import torch.nn as nn

import numpy as np
from PIL import Image
from tqdm import tqdm

from fastai.vision import pil2tensor, ImageSegment, SegmentationLabelList


def get_training_image_size(original_size, multiple=32):
    """
    Our inputs to the network must by multiples of 32.
    We'll find the closest size that both a multiple of 32 and greater than the image size
    """

    new_sizes = []

    for dimension_size in original_size:

        for j in range(20):
            candidate_size = multiple * j
            if candidate_size > dimension_size:
                new_sizes.append(candidate_size)
                break

    if len(new_sizes) != len(original_size):
        raise Exception("Could not find a valid size for the image")

    return tuple(new_sizes)


def multiclass_dice(logits, targets, iou=False, eps=1e-8):
    """
    Dice coefficient metric for multiclass binary target. 
    If iou=True, returns iou metric, classic for segmentation problems.
    """

    n = targets.shape[0]  # Batch size of 4

    # Flatten logits and targets
    logits = logits.view(n, -1)
    targets = targets.view(n, -1).float()

    # Convert logits to probabilities
    probs = torch.sigmoid(logits)
    intersect = (probs * targets).sum(dim=1).float()
    union = (probs + targets).sum(dim=1).float()

    if not iou:
        l = 2. * intersect / union
    else:
        l = intersect / (union - intersect + eps)

    # The Dice coefficient is defined to be 1 when both X and Y are empty.
    # That said, we'd get a divide-by-zero-exception if union was 0 anyways...
    l[union == 0.] = 1.
    return l.mean()


def multiclass_dice_threshold(logits, targets, threshold=0.5, iou=False, eps=1e-8):
    """
    Dice coefficient metric for multiclass binary target. 
    If iou=True, returns iou metric, classic for segmentation problems.
    """

    n = targets.shape[0]  # Batch size of 4

    # Flatten logits and targets
    logits = logits.view(n, -1)
    targets = targets.view(n, -1).float()

    # Convert logits to probabilities
    probs = torch.sigmoid(logits)

    preds = probs
    preds[preds >= threshold] = 1
    preds[preds < threshold] = 0

    intersect = (preds * targets).sum(dim=1).float()
    union = (preds + targets).sum(dim=1).float()

    if not iou:
        l = 2. * intersect / union
    else:
        l = intersect / (union - intersect + eps)

    # The Dice coefficient is defined to be 1 when both X and Y are empty.
    # That said, we'd get a divide-by-zero-exception if union was 0 anyways...
    l[union == 0.] = 1.
    return l.mean()


def override_open_mask():
    # Our masks are overlapping so we've represented the masks as 4-channel images
    # This is convenient for us because we can still store them in standard RGBA images
    # However we have to make sure we load these images as RGBA in order for them to work
    def custom_open_mask(filename, div=False, convert_mode='L', after_open=None):
        x = Image.open(filename).convert('RGBA')
        if after_open:
            x = after_open(x)

        x = pil2tensor(x, np.float32)

        return ImageSegment(x)

    def custom_open(self, fn):
        return custom_open_mask(fn)

    # Open image with our custom method
    SegmentationLabelList.open = custom_open


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
    Returns run length as string formatted
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def post_process(probability, threshold, min_size, shape):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros(shape, np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def convert_mask_to_rle(mask, threshold, min_size):
    preds, num_preds = post_process(mask, threshold, min_size, mask.shape)
    rle_mask = mask2rle(preds)
    return rle_mask


def convert_masks_to_rle(test, test_preds, threshold, min_size):
    unique_test_images = test.iloc[::4, :]

    print(len(unique_test_images))
    for i, row in tqdm(unique_test_images.iterrows()):
        saved_pred = test_preds[i // 4]

        fish_rle = convert_mask_to_rle(saved_pred[0], threshold, min_size)
        flower_rle = convert_mask_to_rle(saved_pred[1], threshold, min_size)
        gravel_rle = convert_mask_to_rle(saved_pred[2], threshold, min_size)
        sugar_rle = convert_mask_to_rle(saved_pred[3], threshold, min_size)

        # Save in dataframe
        test.loc[test['Image_Label'] == row['im_id'] + "_Fish", 'EncodedPixels'] = fish_rle
        test.loc[test['Image_Label'] == row['im_id'] + "_Flower", 'EncodedPixels'] = flower_rle
        test.loc[test['Image_Label'] == row['im_id'] + "_Gravel", 'EncodedPixels'] = gravel_rle
        test.loc[test['Image_Label'] == row['im_id'] + "_Sugar", 'EncodedPixels'] = sugar_rle

    return test


def create_channel_from_rle(encoded_pixels, shape):
    new_channel = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    s = encoded_pixels.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        new_channel[lo:hi] = 1

    new_channel = new_channel.reshape(shape, order='F')
    return new_channel


def create_mask_for_image(df, index):
    fish = df.iloc[index]
    flower = df.iloc[index + 1]
    gravel = df.iloc[index + 2]
    sugar = df.iloc[index + 3]

    full_path = "data/train_images/" + fish['im_id']
    im = Image.open(full_path)

    shape = im.size
    # shape = (1400, 2100)

    fish_channel = np.zeros(shape, dtype=np.uint8)
    flower_channel = np.zeros(shape, dtype=np.uint8)
    gravel_channel = np.zeros(shape, dtype=np.uint8)
    sugar_channel = np.zeros(shape, dtype=np.uint8)

    if isinstance(fish['EncodedPixels'], str):
        fish_channel = create_channel_from_rle(fish['EncodedPixels'], shape)

    if isinstance(flower['EncodedPixels'], str):
        flower_channel = create_channel_from_rle(flower['EncodedPixels'], shape)

    if isinstance(gravel['EncodedPixels'], str):
        gravel_channel = create_channel_from_rle(gravel['EncodedPixels'], shape)

    if isinstance(sugar['EncodedPixels'], str):
        sugar_channel = create_channel_from_rle(sugar['EncodedPixels'], shape)

    # Create fake RGBA image
    new_image = np.stack([fish_channel, flower_channel, gravel_channel, sugar_channel], axis=-1)

    return new_image


def dice_coef(input, target, threshold=None):
    smooth = 1.0
    input_flatten = input.view(-1)
    if threshold is not None:
        input_flatten = (input_flatten > threshold).float()
    target_flatten = target.view(-1)
    intersection = (input_flatten * target_flatten).sum()
    return (
            (2. * intersection + smooth) /
            (input_flatten.sum() + target_flatten.sum() + smooth)
    )


class DiceLoss(nn.Module):
    def __init__(self, log=False):
        super().__init__()
        self.log = log

    def forward(self, input, target):
        dice_coef_value = dice_coef(torch.sigmoid(input), target)
        if self.log:
            return -torch.log(dice_coef_value)
        else:
            return 1 - dice_coef_value


class BCEDiceLoss(nn.Module):
    def __init__(self, log_dice=False):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(log=log_dice)

    def forward(self, input, target):
        target = target.float()
        b_loss = self.bce_loss(input, target)
        d_loss = self.dice_loss(input, target)
        return b_loss + d_loss
