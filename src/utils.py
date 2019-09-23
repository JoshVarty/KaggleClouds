import torch
import numpy as np
from PIL import Image

from fastai.vision import pil2tensor, ImageSegment, SegmentationLabelList

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