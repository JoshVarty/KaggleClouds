import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import cv2

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
    
    n = targets.shape[0]   #Batch size of 4
    
    #Flatten logits and targets
    logits = logits.view(n,-1)  
    targets = targets.view(n,-1).float()
    
    #Convert logits to probabilities
    probs = torch.sigmoid(logits)
    intersect = (probs * targets).sum(dim=1).float()
    union = (probs + targets).sum(dim=1).float()
    
    if not iou: 
        l = 2. * intersect / union
    else: 
        l = intersect / (union-intersect+eps)
        
    # The Dice coefficient is defined to be 1 when both X and Y are empty.
    # That said, we'd get a divide-by-zero-exception if union was 0 anyways...
    l[union == 0.] = 1.
    return l.mean()


def overrideOpenMask():
    # Our masks are overlapping so we've represented the masks as 4-channel images
    # This is convenient for us because we can still store them in standard RGBA images
    # However we have to make sure we load these images as RGBA in order for them to work
    def custom_open_mask(filename, div=False, convert_mode='L', after_open=None):
        x = Image.open(filename).convert('RGBA')
        if after_open: 
            x = after_open(x)
            
        x = pil2tensor(x,np.float32)
            
        return ImageSegment(x)
        

    def custom_open(self, fn):
        return custom_open_mask(fn)
        
    #Open image with our custom method
    SegmentationLabelList.open = custom_open


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

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

codes = np.array(['Fish', 'Flower', 'Gravel', 'Sugar'])

def convertMasksToRle(test):
    unique_test_images = test.iloc[::4, :]

    for i, row in tqdm(unique_test_images.iterrows()):

        predictionId = row['im_id'] + ".npy"
        path = Path("model_predictions")/predictionId
        saved_pred = np.load(path)

        #TODO: Find good values for threshold and min_size
        fishPreds, nFish = post_process(saved_pred[0], 0.5, 10)
        flowerPreds, nFlower = post_process(saved_pred[1], 0.5, 10)
        gravelPreds, nGravel = post_process(saved_pred[2], 0.5, 10)
        sugarPreds, nSugar = post_process(saved_pred[3], 0.5, 10)

        fishRle = mask2rle(fishPreds)
        flowerRle = mask2rle(flowerPreds)
        gravelRle = mask2rle(gravelPreds)
        sugarRle = mask2rle(sugarPreds)

        #Save in dataframe
        test.loc[test['Image_Label'] == row['im_id'] + "_Fish", 'EncodedPixels'] = fishRle
        test.loc[test['Image_Label'] == row['im_id'] + "_Flower", 'EncodedPixels'] = flowerRle
        test.loc[test['Image_Label'] == row['im_id'] + "_Gravel", 'EncodedPixels'] = gravelRle
        test.loc[test['Image_Label'] == row['im_id'] + "_Sugar", 'EncodedPixels'] = sugarRle

    return test
