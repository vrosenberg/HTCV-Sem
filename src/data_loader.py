import numpy
import os
import json

from constants import *

# should return 
def load_data(data_type = "train", percentage_as_str = "100"):
    data = []
    selected_data = open(PREDETERMINED_RANDOM_DATA_PATH,'r')
    
    files = selected_data.read().splitlines()
    if(data_type == "train"):
        files = files[:MAP_PERCENT_TO_AMOUNT[percentage_as_str]]

    for file in files:
        data.append((file,round(calc_average_damage(file, data_type))))
    return data

def calc_average_damage(file, data_type):
    
    label_file_path = os.path.join(os.path.dirname(__file__),"..","dataset",data_type,"labels",file.split('.')[0]+".json")
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

if __name__ == '__main__':
    print(numpy.array(load_data())[:,1])

