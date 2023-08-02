# Take Images from the augmented Images and build the Balanced Dataset. Source data needs to have enough images to fill each class.
import os
import shutil

#-----------------Constants----------------#

# Amount of images per class
LABEL_INSTANCES = 40

# Output path of balanced dataset 
NEW_BALANCED_DATA_PATH  = "./augmented_datasets/balanced_" + str(LABEL_INSTANCES)

# Source path of augmented dataset
AUG_DATA_PATH = "./augmented_datasets/DATASET_NAME"

#------------------------------------------#


LABELS = ["0","1","2","3","4"]

if __name__ == "__main__":

    print("Start data preprocessing ...")

    if(not os.path.exists(NEW_BALANCED_DATA_PATH)):
            os.makedirs(NEW_BALANCED_DATA_PATH)  

    for label in  LABELS:
        aug_data = os.listdir(AUG_DATA_PATH + "/" + label)
        dest_path = NEW_BALANCED_DATA_PATH + "/" + label
        if(not os.path.exists(dest_path)):
            os.makedirs(dest_path)  
        for img in aug_data[:LABEL_INSTANCES]:
            source_file = AUG_DATA_PATH + "/" + label + "/" + img
            shutil.copy2(source_file, dest_path)

    print("Finished.")
    
