# HTCV-Sem
Baseline and modified model of xView2 challenge for a Computer Vision Seminar

USING OPENCV 4.7.0


Usage:

download xview2 data

put the hold, test and train folder into a data folder

put xview2.txt inside the data folder

do:

make a ordered dataset for the baseline classifier and mixing dataset and balancing dataset

python3.10 src/data_preprocessor.py -src ./data -out ./dataset  


perform ia
mode: random
-> source folder should be the unaltered unorganized data
-> out folder defines the augmented image directory
-> set datasize to the size of the original data pool i.e. 280, 700, 1400, 2800
-> set number to the amount of images to sample, for further steps to work we recommend to sample atleast 1000
-> type 'python3.10 src/in_augment.py -h' for help

python3.10 src/in_augment.py -src "./data/train/" -out "./augmented_datasets/DATASET_NAME/" -ds 280 -num 1000

mode: single-instance/linear
-> source folder should be the unaltered unorganized data
-> out folder defines the augmented image directory
-> set datasize to the size of the original data pool i.e. 280, 700, 1400, 2800
-> type 'python3.10 src/in_augment.py -h' for help
-> performs ia once per image, resulting augmented dataset is as large as datasize

python3.10 src/in_augment.py -src "./data/train/" -out "./augmented_datasets/DATASET_NAME/" -ds 280


mix datasets
-> used for half-half, majority original, majority augmented
-> go into augmented_data_preprocessor.py and change constants as wanted. Consider that both source datasets need to have enough images
-> use as source the from the first step and the one from the mode random step

python3.10 src/augmented_data_preprocessor.py  




python3.10 src/balanced_data_preprocessor.py

