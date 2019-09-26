import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

from src.utils import convertMasksToRle, create_mask_for_image, convert_mask_to_rle


class TestLossesAgainstDetectron(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.DATA = Path("data")
        train = pd.read_csv("data/train.csv")
        train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
        train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
        train = train.fillna('')

        cls.df = train
        print(cls.df.head())
        print(cls.df.head())

    def test(self):
        #open original RLE
        index = 0
        im_id = self.df.iloc[index + 0]['im_id']
        rle_1 = self.df.iloc[index + 0]['EncodedPixels']
        rle_2 = self.df.iloc[index + 1]['EncodedPixels']
        rle_3 = self.df.iloc[index + 2]['EncodedPixels']
        rle_4 = self.df.iloc[index + 3]['EncodedPixels']

        #open original image
        path = self.DATA/'train_images'/im_id
        img = Image.open(path)
        width, height = img.size

        #convert to mask
        mask = create_mask_for_image(self.df, index)

        #convert back to RLE
        fishRle = convert_mask_to_rle(mask[:,:,0], 0.5, 1)
        flowerRle = convert_mask_to_rle(mask[:,:,1], 0.5, 1)
        gravelRle = convert_mask_to_rle(mask[:,:,2], 0.5, 1)
        sugarRle = convert_mask_to_rle(mask[:,:,3], 0.5, 1)

        #compare
        self.assertEqual(rle_1, fishRle)
        self.assertEqual(rle_2, flowerRle)
        self.assertEqual(rle_3, gravelRle)
        self.assertEqual(rle_4, sugarRle)
    




if __name__ == '__main__':
    unittest.main()
