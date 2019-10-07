# KaggleClouds
Working on: https://www.kaggle.com/c/understanding_cloud_organization

This repository is comprised of exploratory notebooks and a single training script (`src/main.py`)

## Description

- **00_EDA.ipynb** Contains a short EDA and generates masks on disk from the run-length encodings
- **01_BasicModel.ipynb** Demonstrates a simple model
- **02_InvestigateImages.ipynb** Explores problems in the dataset and interesting images
- **04_ImageSimilarity_GetData.ipynb** Gets data that we will use to train a SiameseNetwork
- **04_ImageSimilarity_SiameseNetwork.ipynb** Trains a SiameseNetwork to find similar images

## Usage
In order to use this repository:

1. Run all cells in `00_EDA.ipynb` to generate the masks
2. Run `src/main.py` to produce a submission

## Folder Structure
 - `/data` contains all training data
 - `/model_source` contains the exact `main.py` source code that lead to a given valid score
 - `/submissions` contains the generated submission `.csv` file for a given valid score
 - `/tests` a handful of unit tests for run-length encoding conversions
 
 
 

