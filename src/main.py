import os
import torch
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from fastai.vision import SegmentationItemList, imagenet_stats, get_transforms, models
from fastai.vision import unet_learner, BCEWithLogitsFlat, DatasetType, get_preds, load_learner

from utils import multiclass_dice, overrideOpenMask

NFOLDS = 2
RANDOM_STATE = 42
skf = StratifiedKFold(n_splits=NFOLDS, random_state=RANDOM_STATE)

script_name = os.path.basename(__file__).split('.')[0]
MODEL_NAME = "{0}__folds{1}".format(script_name, NFOLDS)
print("Working dir", os.getcwd())
print("Model: {}".format(MODEL_NAME))

# Make required folders if they're not already present
directories = ['./kfolds', './model_predictions', './model_source']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)


DATA = Path('data')
TRAIN = DATA/"train.csv"
TEST = DATA/"sample_submission.csv"

#If we want to train on smaller images, we can add their suffix here
size = (350,525)
SUFFIX = "_" + str(size[0]) + "x" + str(size[1])        #eg. _350x525
batch_size=8

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
test['label'] = test['Image_Label'].apply(lambda x: x.split('_')[1])
test['im_id'] = test['Image_Label'].apply(lambda x: x.split('_')[0])

unique_images = train.iloc[::4, :]

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

# Create empty predictions for each of the test images
# We have to save them to disk because we don't have enough memory :(
unique_test_images = test.iloc[::4, :]
for index, row in unique_test_images.iterrows():
    im_id = row["im_id"]
    empty_preds = np.zeros((len(codes), size[0], size[1]))
    path = Path("model_predictions")/im_id
    np.save(path, empty_preds)

i = 0
for train_index, valid_index in skf.split(id_mask_count['img_id'].values, id_mask_count['count']):

    src = (SegmentationItemList.from_df(unique_images, DATA/('train_images'+str(SUFFIX)), cols='im_id')
        .split_by_idx(valid_index)
        .label_from_func(get_y_fn, classes=codes))

    #TODO: Fix these
    transforms = get_transforms()

    data = (src.transform(transforms, tfm_y=True, size=size)
            .databunch(bs=batch_size)
            .normalize(imagenet_stats))

    learn = unet_learner(data, models.resnet18, metrics=[multiclass_dice], loss_func=BCEWithLogitsFlat(), model_dir=DATA)

    learn.fit_one_cycle(1, 1e-3)
    #learn.unfreeze()
    #learn.fit_one_cycle(1, slice(1e-6, 1e-3))
    #valid_dice_score = learn.recorder.metrics[-1]
    #all_dice_scores.append(valid_dice_score)

    #Save model
    filename = MODEL_NAME + '_' + str(i)
    learn.export(learn.export(file=filename))

    # Generate test predictions, chunk-by-chunk based on this single fold
    numItems = (len(unique_test_images) + 1) // 10
    for i in range(10):
        start = i * numItems
        end = min((i + 1) * numItems, len(unique_test_images) - 1)

        test_src = SegmentationItemList.from_df(unique_test_images[start:end], DATA/('test_images'+str(SUFFIX)), cols='im_id')
        learn = load_learner(filename, test=test_src)
        preds, y = get_preds(ds_type=DatasetType.Test)
        preds = preds / NFOLDS

        for i, row in unique_test_images[start:end].iterrows():

            im_id = row['im_id']
            current_pred = preds[i]

            path = Path("model_predictions")/im_id
            saved_pred = np.load(path)
            saved_pred = saved_pred + current_pred
            np.save(path, saved_pred)

    i = i + 1

score = np.mean(all_dice_scores)
print("Total Dice", score)

print("Saving code...")
shutil.copyfile(os.path.basename(__file__), 'model_source/{}__{}.py'.format(MODEL_NAME, str(score)))



