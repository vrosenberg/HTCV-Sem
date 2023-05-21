import os
import shutil
from data_loader import *


#print(train_data_entries)
def format_hold_data():
    data = []
    files = os.listdir(os.path.join(os.path.dirname(__file__),"..","data","hold","images"))
    for file in files:
        if(file.find("post") >= 0) :
            data.append((file,round(calc_average_damage(file, "hold"))))

    source_path = os.path.join(os.path.dirname(__file__),"..","data","hold","images")  # Path to the source file
    destination_path = os.path.join(os.path.dirname(__file__),"..","dataset","hold")  # Path to the destination directory

    idx = 0
    for file, label in data:
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
        
        #print(source_file)
        #print(destination_file)
        shutil.copy2(source_file, destination_file)
        idx += 1 

def format_test_data():
    data = []
    
    files = os.listdir(os.path.join(os.path.dirname(__file__),"..","data","test","images"))
    for file in files:
        if(file.find("post") >= 0) :
            data.append((file,round(calc_average_damage(file, "test"))))
    
    source_path = os.path.join(os.path.dirname(__file__),"..","data","test","images")  # Path to the source file
    destination_path = os.path.join(os.path.dirname(__file__),"..","dataset","test")  # Path to the destination directory

    idx = 0
    for file, label in data:
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
        
        print(source_file)
        print(destination_file)
        #shutil.copy2(source_file, destination_file)
        idx += 1 
    print(idx)


def format_train_data():
    data = []
    
    files = os.listdir(os.path.join(os.path.dirname(__file__),"..","data","train","images"))
    for file in files:
        if(file.find("post")>=0):
            data.append((file,round(calc_average_damage(file, "train"))))

    source_path = os.path.join(os.path.dirname(__file__),"..","data","train","images")  # Path to the source file
    destination_path = os.path.join(os.path.dirname(__file__),"..","dataset","train")  # Path to the destination directory

    idx = 0
    for file, label in data:
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
        
        print(source_file)
        print(destination_file)
        
        shutil.copy2(source_file, destination_file)
        idx += 1    
if __name__ == '__main__':
    format_train_data()