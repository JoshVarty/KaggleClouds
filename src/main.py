import os
import torch
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from fastai.vision import SegmentationItemList, imagenet_stats, get_transforms, models
from fastai.vision import unet_learner, BCEWithLogitsFlat, DatasetType, load_learner
from fastai.vision import ResizeMethod, EmptyLabel
from tqdm import tqdm
from functools import partial
from utils import multiclass_dice, overrideOpenMask, get_training_image_size
from utils import convertMasksToRle, post_process

def multiclass_dice_threshold(logits, targets, threshold=0.5, iou=False, eps=1e-8):
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
    
    preds = probs
    preds[preds >= threshold] = 1
    preds[preds < threshold] = 0
    
    intersect = (preds * targets).sum(dim=1).float()
    union = (preds + targets).sum(dim=1).float()
    
    if not iou: 
        l = 2. * intersect / union
    else: 
        l = intersect / (union-intersect+eps)
        
    # The Dice coefficient is defined to be 1 when both X and Y are empty.
    # That said, we'd get a divide-by-zero-exception if union was 0 anyways...
    l[union == 0.] = 1.
    return l.mean()

NFOLDS = 2
RANDOM_STATE = 42
skf = StratifiedKFold(n_splits=NFOLDS, random_state=RANDOM_STATE)

script_name = os.path.basename(__file__).split('.')[0]
MODEL_NAME = "{0}__folds{1}".format(script_name, NFOLDS)
print("Working dir", os.getcwd())
print("Model: {}".format(MODEL_NAME))

# Make required folders if they're not already present
directories = ['./kfolds', './model_predictions', './model_source', './submissions']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

DATA = Path('data')
TRAIN = DATA/"train.csv"
TEST = DATA/"sample_submission.csv"

size = (350,525)
training_image_size = get_training_image_size(size)     #UNet requires that inputs are multiples of 32
#If we want to train on smaller images, we can add their suffix here
SUFFIX = "_" + str(size[0]) + "x" + str(size[1])        #eg. _350x525
batch_size=8

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
train = train.iloc[:4000]
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
test['label'] = test['Image_Label'].apply(lambda x: x.split('_')[1])
test['im_id'] = test['Image_Label'].apply(lambda x: x.split('_')[0])

unique_images = train.iloc[::4, :]
unique_test_images = test.iloc[::4, :]

#Ensure we open our 4D masks properly
overrideOpenMask()

def get_y_fn(x):
    # Given a path to a training image, build the corresponding mask path
    split = x.split('/')
    newPath = DATA/("train_images_annots" + SUFFIX)/split[-1].replace('.jpg','.png')
    return newPath

codes = np.array(['Fish', 'Flower', 'Gravel', 'Sugar'])

# We count how many non-NAN labels are present for each image
# Then we create our K-Fold splits based on this
id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})

all_dice_scores = []

test_preds = np.zeros((len(unique_test_images), len(codes), size[0], size[1]))

#Loss metrics
dice_25 = partial(multiclass_dice_threshold, threshold=0.25)
dice_50 = partial(multiclass_dice_threshold, threshold=0.50)
dice_75 = partial(multiclass_dice_threshold, threshold=0.75)

currentFold = 0
for train_index, valid_index in skf.split(id_mask_count['img_id'].values, id_mask_count['count']):

    src = (SegmentationItemList.from_df(unique_images, DATA/('train_images'+str(SUFFIX)), cols='im_id')
        .split_by_idx(valid_index)
        .label_from_func(get_y_fn, classes=codes))

    transforms = get_transforms(max_warp=0, max_rotate=0)
    data = (src.transform(transforms, tfm_y=True, size=training_image_size, resize_method=ResizeMethod.PAD, padding_mode="zeros")
            .databunch(bs=batch_size)
            .normalize(imagenet_stats))


    learn = unet_learner(data, models.xresnet18, pretrained=False, metrics=[multiclass_dice, dice_25, dice_50, dice_75], loss_func=BCEWithLogitsFlat(), model_dir=DATA)

    learn.fit_one_cycle(10, 1e-3)
    learn.unfreeze()
    learn.fit_one_cycle(30, slice(1e-6, 1e-3))
    valid_dice_score = learn.recorder.metrics[-1]
    print("DEBUG", valid_dice_score)
    all_dice_scores.append(valid_dice_score)

    #Save model
    filename = MODEL_NAME + '_' + str(currentFold)
    print(filename)
    learn.export()

    # Generate test predictions, chunk-by-chunk based on this single fold
    numberOfBatches = (len(unique_test_images) + 1) // batch_size

    for i in tqdm(range(numberOfBatches))   :
        start = i * batch_size
        end = min((i + 1) * batch_size, len(unique_test_images) - 1)

        test_src = SegmentationItemList.from_df(unique_test_images[start:end], DATA/('test_images'+str(SUFFIX)), cols='im_id')

        learn = load_learner(DATA/('train_images'+str(SUFFIX)), test=test_src, tfm_y=False)
        preds, _ = learn.get_preds(ds_type=DatasetType.Test)
        preds = preds / NFOLDS

        for j, current_pred in enumerate(preds):

            currentIndex = start + j
            row = unique_test_images.iloc[currentIndex]
            predictionId = row['im_id'] + ".npy"

            # We have to correct for the resizing/padding of our original images
            # To do this, we take a crop of our prediction in the proper size
            valid_pred = current_pred[:, :size[0], :size[1]]
            test_preds[j] = test_preds[j] + valid_pred.numpy()

    currentFold = currentFold + 1

score = np.mean(all_dice_scores)
print("Total Dice", score)

print("Saving code...")
shutil.copyfile(__file__, 'model_source/{}__{}.py'.format(MODEL_NAME, str(score)))

#Generate the submission
submission = convertMasksToRle(test, test_preds, threshold=0.5, min_size=10000)
submission = submission.drop(columns=['label', 'im_id'])
submission.to_csv("submissions/{}__{}.csv".format(filename, str(score)), index=False)








