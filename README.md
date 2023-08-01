# Hot Topics in Computer Vision
## Addressing Label Scarcity in the Semantic Analysis of Earth Observation Imagery

We implement the data augmentation method InAugment by Arar et al.

## What does this project contain
- Preprocessing scripts
- InAugment implementation
- Notebooks for classifiers


## Requirements

PYTHON 3.10

## Installation
```
pip install -r requirements.txt
```
## Usage

1. Download [xview2](https://xview2.org/) train, test and holdout data.

2. Create a *data* directory in the root directory and put all three into the *data* directory.

3. Put the xview2.txt inside the data folder

### Make an ordered dataset
This takes the raw images from the xview2 data and orders them into the required directory structure. Type ```python3 src/data_preprocessor.py -h``` for help.
This also equals to the *Baseline* dataset
```
python3 src/data_preprocessor.py -src ./data -out ./dataset  
```
### Perform InAugment
**mode**: random  

Generates a specified amount of augmented images from a given original data pool.
- type ```python3 src/in_augment.py -h``` for help
- to change InAugment hyperparameters change CONSTANTS in inaugment.py
- source folder should be the unaltered unorganized data
- output folder defines the augmented image directory
- set datasize to the size of the original data pool i.e. 280, 700, 1400, 2800
- set number to the amount of images to sample, for further steps a certain amount of images need to be generated

```
python3 src/in_augment.py -src "./data/train/" -out "./augmented_datasets/DATASET_NAME/" -ds 280 -num 1000
```

**mode**: single-instance/linear   

This generates the dataset for the single-instance/linear model
- type ```python3 src/in_augment.py -h``` for help
- to change InAugment hyperparameters change CONSTANTS in inaugment.py
- source folder should be the unaltered unorganized data
- output folder defines the augmented image directory
- set datasize to the size of the original data pool i.e. 280, 700, 1400, 2800
- performs InAugment once per image, resulting augmented dataset is as large as datasize
  
```
python3 src/in_augment.py -src "./data/train/" -out "./augmented_datasets/DATASET_NAME/" -ds 280
```

### Mix datasets
Combine regular and augmented data to create *half-half*, *majority original*, *majority augmented* dataset
- go into augmented_data_preprocessor.py and change constants as wanted. Consider that both source datasets need to have enough images
- original data path points to the regular ordered images
- augmented data path points to the augmented images
- new mixed path is the output path of the mixed dataset
- original data size refers to the size of the original data pool, i.e. 280, 700, etc.
- augmented data to original data ratio describes how many augmented images are being placed into the dataset in relation to the data size. Ratio = 2.0 => twice amount of augmented images than regular/original

```
python3.10 src/augmented_data_preprocessor.py  
```

### Balance datasets
Take images from the augmented images and build the *balanced* dataset. Source data needs to have enough images to fill each class.
- go into balanced_data_preprocessor.py and change constants as wanted. Consider that both source datasets need to have enough images
- augmented data path are the source images we use to balance the dataset with
- new balanced path is the output path of the balanced dataset
- label instances refers to the amount of images per class

```
python3.10 src/balanced_data_preprocessor.py
```

## Average Building Damage Classification

Under *notebooks/baseline.ipynb* is a jupyter notebook for the *baseline* model.

Under *notebooks/inAugment.ipynb* is a jupyter notebook that trains on any given model.

The datasets for the models need to be in the structure:

dataset_dir:
```
.
├── 0
├── 1
├── 2
├── 3
└── 4
```

Where the numbers represent different JDS labels:
- 1: No Damage
- 2: Minor Damage
- 3: Major Damage
- 4: Destroyed
- 0: No Buildings (Misc/Filler label)




