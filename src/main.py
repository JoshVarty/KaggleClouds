import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold

print(os.getcwd())
from utils import multiclass_dice, overrideOpenMask

from fastai.vision import SegmentationItemList, imagenet_stats, get_transforms, models
from fastai.vision import unet_learner, BCEWithLogitsFlat

# Make required folders if they're not already present
directories = ['./kfolds', './model_predictions', './model_source']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

NFOLDS = 5
RANDOM_STATE = 42
skf = StratifiedKFold(n_splits=NFOLDS, random_state=RANDOM_STATE)

DATA = Path('data')
TRAIN = DATA/"train.csv"
TEST = DATA/"test.csv"

#If we want to train on smaller images, we can add their suffix here
size = (350,525)
SUFFIX = "_" + str(size[0]) + "x" + str(size[1])        #eg. _350x525
batch_size=8

train = pd.read_csv(TRAIN)
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

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
    learn.unfreeze()
    learn.fit_one_cycle(10, slice(1e-6, 1e-3))





