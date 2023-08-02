# Mixes data samples of augmented and not augmented data
import os
import shutil
import cv2
import argparse

#---------------CONSTANTS-----------------#

# Output path of mixed dataset
NEW_MIXED_DATA_PATH  = "./augmented_datasets/mixed_dataset"

# Path to original/regular images directory (already ordered)
ORIG_DATA_PATH = "./dataset/train"

# Path to augmented image directory
AUG_DATA_PATH = "./augmented_datasets/DATASET_NAME"

# Path to xview2.txt file
XVIEW2_TXT = "./data/xview2.txt"

# Dataset size of original image pool. i.e. 280, 700, 1400, 2799
ORIG_DS_SIZE = 280

# Ratio of of augmented to regular images. ratio = 2.0 => twice as many augmented images as original images
AUG_DATA_TO_ORIG_RATIO = 0.5

# Resizing high resolution xview2 images to resnet input size to save memory
RESNET_INPUT_SHAPE = (224,224)

#-----------------------------------------#

# Mixes Augmented and Original Images into a Dataset
def preprocess_data():

    with open(XVIEW2_TXT,"r") as file:
        xview_entries = file.readlines()
    xview_entries = [line.strip() for line in xview_entries]
    xview_entries = xview_entries[:ORIG_DS_SIZE]

    if(not os.path.exists(NEW_MIXED_DATA_PATH)):
            os.makedirs(NEW_MIXED_DATA_PATH)  

    
    orig_data_root = os.listdir(ORIG_DATA_PATH)
    orig_data_label_instances = {
         "0": 0,
         "1": 0,
         "2": 0,
         "3": 0,
         "4": 0,
    }
    for label in orig_data_root:
        orig_data = os.listdir(ORIG_DATA_PATH + "/" +label)
        dest_path = NEW_MIXED_DATA_PATH + "/" + label
        if(not os.path.exists(dest_path)):
            os.makedirs(dest_path)  
        
        for img in orig_data:
             if(img in xview_entries):
                orig_data_label_instances[label] += 1
                source_file = ORIG_DATA_PATH + "/" + label + "/" + img
                image = cv2.imread(source_file, cv2.IMREAD_UNCHANGED)
                image = cv2.resize(image, RESNET_INPUT_SHAPE)
                dst_file = dest_path + "/" + img
                cv2.imwrite(dst_file,image)
    print("orig")
    print(orig_data_label_instances)

    aug_data_label_instances = {}
    for label,count in orig_data_label_instances.items():
        aug_data = os.listdir(AUG_DATA_PATH + "/" + label)
        aug_data_label_instances[label]= round(orig_data_label_instances[label]*AUG_DATA_TO_ORIG_RATIO)
        dest_path = NEW_MIXED_DATA_PATH + "/" + label
        for i in range(aug_data_label_instances[label]):
            source_file = AUG_DATA_PATH + "/" + label + "/" + aug_data[i]
            shutil.copy2(source_file,dest_path)
    print("aug")
    print(aug_data_label_instances)

if __name__ == "__main__":

    print("Start data preprocessing ...")

    preprocess_data()

    print("Finished.")
