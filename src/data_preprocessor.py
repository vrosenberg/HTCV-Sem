import os
import json
import shutil
import argparse

MAP_PERCENT_TO_AMOUNT = {
    "10" : 280,
    "25" : 700,
    "50" : 1400,
    "100": 2799
}

DAMAGE_LEVEL_TO_SCORE = {
    "destroyed" : 4,
    "major-damage" : 3,
    "minor-damage" : 2,
    "no-damage" : 1,
    "no-building" : 0
}

'''
def load_data_entries(path_to_xview2txt, data_type = "train", percentage_as_str = "100"):
    data = []
    selected_data = open(path_to_xview2txt,'r')
    
    files = selected_data.read().splitlines()
    if(data_type == "train"):
        files = files[:MAP_PERCENT_TO_AMOUNT[percentage_as_str]]

    for file in files:
        data.append((file,round(calc_average_damage(file, data_type))))
    return data
'''

def calc_average_damage(source_path, file, data_type):
    
    label_file_path = os.path.join(source_path,data_type,"labels",file.split('.')[0]+".json")
    f = open(label_file_path)
    label_data = json.load(f)
    total_dmg = 0.0
    count = 0.0
    for polygon in label_data["features"]["xy"]:
        
        if(polygon["properties"]["feature_type"] == 'building' and polygon["properties"]["subtype"] != 'un-classified'):
            count += 1.
            total_dmg += DAMAGE_LEVEL_TO_SCORE[polygon["properties"]["subtype"]]
        if(polygon["properties"]["feature_type"] != 'building'):
            print("NO BUILDING")
    if(count == 0):
        return 0
    return total_dmg/count

def format_data(source_dir, output_dir, data_partition):
    files = os.listdir(os.path.join(source_dir,data_partition,"images"))
    data = [(file, round(calc_average_damage(source_dir, file, data_partition))) for file in files if(file.find("post") >= 0)]

    source_path = os.path.join(os.path.join(source_dir,data_partition,"images"))
    destination_path = os.path.join(output_dir,data_partition)  # Path to the destination directory
    
    for idx, (file, label) in enumerate(data):
        source_file = os.path.join(source_path,file)
        
        if(label == 0):
            destination_file = os.path.join(destination_path,"0")
        elif( label == 1):
            destination_file = os.path.join(destination_path,"1")
        elif( label == 2):
            destination_file = os.path.join(destination_path,"2")
        elif( label == 3):
            destination_file = os.path.join(destination_path,"3")
        elif( label == 4):
            destination_file = os.path.join(destination_path,"4")

        if not os.path.exists(destination_file):
            os.makedirs(destination_file)
        shutil.copy2(source_file, destination_file)

if __name__ == '__main__':
    print("Start data preprocessing ...")

    parser = argparse.ArgumentParser(description='Rearranges the xview2 dataset structure to fit the folder structure specified by pytorch\'s Image Dataloader.')
    parser.add_argument('-src', '--source', type=str, help='Source directory of dataset', required=True)
    parser.add_argument('-out', '--output', type=str, help='Output directory of rearranged dataset', required=True)
    args = parser.parse_args()
    source_dir = args.source #'./data'
    output_dir = args.output #'./dataset'

    data_partitions = ["train", "test", "hold"]

    for data_partition in data_partitions:
        format_data(source_dir, output_dir,data_partition)

    print("Finished.")
    