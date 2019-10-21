import os
import gc
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image
from functools import partial
from sklearn.model_selection import train_test_split

import torch

import fastai
from fastai.torch_core import grab_idx
from fastai.core import subplots, split_kwargs_by_func
from fastai.basic_train import RecordOnCPU
from fastai.layers import BCEWithLogitsFlat
from fastai.vision import unet_learner, models, DatasetType
from fastai.vision import lr_find
from fastai.vision import Path, get_image_files, open_image, open_mask, SegmentationItemList, get_transforms, \
    imagenet_stats
from fastai.metrics import dice
from fastai.vision import ResizeMethod
from fastai.vision import ImageSegment, SegmentationLabelList, pil2tensor

from utils import multiclass_dice, multiclass_dice_probs, multiclass_dice_threshold, get_training_image_size, rle_decode, \
    override_open_mask
from utils import BCEDiceLoss
from utils import post_process, convert_masks_to_rle

RANDOM_STATE = 42

script_name = os.path.basename(__file__).split('.')[0]
MODEL_NAME = script_name

if os.getcwd().endswith('src'):
    # We want to be working in the root directory
    os.chdir('../')

print("Working dir", os.getcwd())
print("Model: {}".format(MODEL_NAME))

DATA = Path('data')
TRAIN = DATA / "train_split.csv"
TEST = DATA / "sample_submission.csv"

size = (350, 525)
training_image_size = get_training_image_size(size)  # UNet requires that inputs are multiples of 32
# If we want to train on smaller images, we can add their suffix here
SUFFIX = "_" + str(size[0]) + "x" + str(size[1])  # eg. _350x525
batch_size = 8

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
test['label'] = test['Image_Label'].apply(lambda x: x.split('_')[1])
test['im_id'] = test['Image_Label'].apply(lambda x: x.split('_')[0])

unique_images = train.iloc[::4, :]
unique_test_images = test.iloc[::4, :]

# Our masks were converted from four run-length encodings into 4-channel masks.
# For the sake of convenience, I've saved these masks as RGBA .png images on disks
# Use a custom approach for opening masks as RGBA
override_open_mask()

def get_y_fn(x):
    # Given a path to a training image, build the corresponding mask path
    split = x.split('/')
    new_path = DATA / ("train_images_annots" + SUFFIX) / split[-1].replace('.jpg', '.png')
    return new_path

codes = np.array(['Fish', 'Flower', 'Gravel', 'Sugar'])

id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(
    lambda x: x.split('_')[0]).value_counts(). \
    reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=RANDOM_STATE,
                                        stratify=id_mask_count['count'], test_size=0.1)

test_src = SegmentationItemList.from_df(unique_test_images, DATA / ('test_images' + str(SUFFIX)), cols='im_id')

src = (SegmentationItemList.from_df(unique_images, DATA / 'train_images_350x525', cols='im_id')
       .split_from_df(col='Valid')
       .label_from_func(get_y_fn, classes=codes))


transforms = get_transforms()
data = (src.transform(transforms, tfm_y=True, size=training_image_size, resize_method=ResizeMethod.PAD,
                      padding_mode="zeros")
        .add_test(test_src, tfm_y=False)
        .databunch(bs=batch_size)
        .normalize(imagenet_stats))

dice_50 = partial(multiclass_dice_threshold, threshold=0.50)

learn = unet_learner(data, models.resnet18, pretrained=True, metrics=[multiclass_dice, dice_50],
                     loss_func=BCEWithLogitsFlat(), model_dir=DATA)

# learn.fit_one_cycle(10, 1e-3)
# learn.unfreeze()
# learn.fit_one_cycle(30, slice(1e-6, 1e-3))
# valid_dice_score = learn.recorder.metrics[-1]
# print("Raw Valid Score", valid_dice_score)

# Get the raw predictions and targets for our validation dataset
preds, targets = learn.get_preds(DatasetType.Valid)

if preds.max() > 1:
    # If we use custom loss functions, we have to apply the activation ourselves
    print("VALID: It looks like these are logits. Max:", preds.max())
    preds = torch.sigmoid(preds)
preds = preds.numpy()
preds = preds[:, :, :350, :525]
targets = targets[:, :, :350, :525].contiguous()

# Convert the raw predictions to thresholded predictions
threshold = 0.5
min_size = 10000
for i in range(len(preds)):
    current_pred = preds[i]

    fish_preds, _ = post_process(current_pred[0], threshold, min_size)
    flower_preds, _ = post_process(current_pred[1], threshold, min_size)
    gravel_preds, _ = post_process(current_pred[2], threshold, min_size)
    sugar_preds, _ = post_process(current_pred[3], threshold, min_size)

    preds[i][0] = fish_preds
    preds[i][1] = flower_preds
    preds[i][2] = gravel_preds
    preds[i][3] = sugar_preds

# Get the dice score for our threshold predictions
score = multiclass_dice_probs(torch.Tensor(preds).contiguous(), targets)
np.save("preds", preds)
print("Threshold Validation Dice", score)

print("Saving code...")
shutil.copyfile(__file__, 'model_source/{}__{}.py'.format(MODEL_NAME, str(score)))

# Get test predictions
test_preds, _ = learn.get_preds(DatasetType.Test)
if test_preds.max() > 1:
    # If we use custom loss functions, we have to apply the activation ourselves
    print("TEST: It looks like these are logits. Max:", test_preds.max())
    test_preds = torch.sigmoid(test_preds)

test_preds = test_preds.numpy()
test_preds = test_preds[:, :, :350, :525]

submission = convert_masks_to_rle(test, test_preds, threshold=0.5, min_size=10000)

submission = submission.drop(columns=['label', 'im_id'])
submission.to_csv("submissions/{}__{}.csv".format(script_name, str(score)), index=False)
