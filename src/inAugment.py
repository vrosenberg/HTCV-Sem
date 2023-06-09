import cv2
import numpy as np
import json
import re
import os
from data_loader import calc_average_damage
from shapely.geometry import Point, Polygon



# returns [(y,h,x,w,label), ...]
def get_building_patches(json_path):
    f = open(json_path)
    label_data = json.load(f)
    patches = []
    for polygon in label_data["features"]["xy"]:
        
        if(polygon["properties"]["feature_type"] == 'building' and polygon["properties"]["subtype"] != 'un-classified'):
            #print(polygon["wkt"].split(',') )
            points = re.findall(r'\d+\.\d+',polygon["wkt"])
            x_coords = [float(point) for point in points[::2]]
            y_coords = [float(point) for point in points[1::2]]
            
            y = round(min(y_coords))
            h = round(max(y_coords) - min(y_coords))
            x = round(min(x_coords))
            w = round(max(x_coords) - min(x_coords))
            label = polygon["properties"]["subtype"]
            patches.append((y,h,x,w,label))
    return patches

def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop

'''
def get_random_patch(image, H_patch = 32, W_patch = 32):
    
    patch_size = [W_patch, H_patch]

    min_x = 1 - patch_size[0] 
    min_y = 1 - patch_size[1]

    max_x = image.shape[0] - 1 
    max_y = image.shape[1] - 1

    x = np.random.randint(min_x,max_x+1)
    y = np.random.randint(min_y,max_y+1)

    #trim patch_size according to random crop pos
    actual_patch_size = [patch_size[0],patch_size[1]]
    if(x < 0):
        actual_patch_size[0] = x + patch_size[0]
        x = 0
    if(y < 0):
        actual_patch_size[1] = y + patch_size[1]
        y = 0
    if(x + patch_size[0] > image.shape[0]):
        actual_patch_size[0] = image.shape[0] - x
    if(y + patch_size[1] > image.shape[1]):
        actual_patch_size[1] = image.shape[1] - y

    patch = image[y:y+patch_size[1],x:x+patch_size[0]]

    return patch
'''

def get_random_patch(image, H_patch = 128, W_patch = 128):
    
    patch_size = [W_patch, H_patch]

    min_x = 0
    min_y = 0

    max_x = image.shape[0] - patch_size[0]
    max_y = image.shape[1] - patch_size[1]

    x = np.random.randint(min_x,max_x+1)
    y = np.random.randint(min_y,max_y+1)

    patch = image[y:y+patch_size[1],x:x+patch_size[0]]

    return patch


if __name__ == "__main__":
    data_type = "train"
    #path = 'hurricane-florence_00000163_post_disaster.png'
    name = 'guatemala-volcano_00000000_post_disaster.png'
    #print(calc_average_damage(name,data_type=data_type))
    path = 'data/train/images/'+name
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    #get_copied_patch(path)
    #image = np.zeros((5,5))
    
    #take random patch+
    #(x,y)
    #patch = get_random_patch(image) 
    json_names = os.listdir("data/train/labels")

    for json_name in json_names:
        if("pre" in json_name):
            continue
        json_path = "./data/" + "train" + "/labels/" + json_name
        f = open(json_path)
        label_data = json.load(f)
        polygons = label_data['features']['xy']
        for building in polygons:
            polygon_str = building['wkt']
            polygon_points = []
            points_str = polygon_str.replace('(', "").replace(')', "").replace("POLYGON ","").replace(',',"")
            points = [float(point) for point in points_str.split(' ')]

            for i in range(0,len(points),2):
                polygon_points.append((points[i],points[i+1]))
            Polygon(polygon_points)
            
    'POLYGON ((375.8542572463341 1024, 375.7982769199857 1021.698346053432, 386.2017761306936 1021.732499438006, 386.1684428017282 1024, 375.8542572463341 1024))'
    #trim_patch_size = 
    print("END")
    '''
    json_path = "./data/" + data_type + "/labels/" + name.split('.')[0]+".json"

    patches = get_building_patches(json_path=json_path)

    path = 'data/train/images/'+name
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    for (y,h,x,w,label) in patches:
        mask = np.zeros([h,w,image.shape[2]], dtype=np.uint8)
        image[y:y+h,x:x+w] = mask


    #'POLYGON ((917.7720847921884 732.7062537792558, 907.6213628757642 725.1242804382573, 914.9839923437013 715.9347330488779, 925.5281487485797 722.1464920387161, 922.3015423879609 727.7179381319243, 917.7720847921884 732.7062537792558))'



    cv2.imshow('Result', image)
    cv2.waitKey(0)

    #cv2.imwrite("augmented"+name,image)

    '''
    '''
    json_path = "./data/" + data_type + "/labels/" + name.split('.')[0]+".json"

    patches = get_building_patches(json_path=json_path)
    print(patches)
    path = 'src/lenna.png'

    # Load the PNG image
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    resized_img = cv2.resize(image,(244,244))
    cv2.imshow('Result', resized_img)
    cv2.waitKey(0)


    # Check if the image was loaded successfully
    if image is not None:
        # Display the image
        cv2.imshow('PNG Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('Failed to load the image.')

    y = 200
    h = 200
    x = 50
    w = 80
    crop_img = image[y:y+h, x:x+w].copy()
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)


    # Set the position where you want to place the smaller image in the larger image
    x_pos = 100  # x-coordinate of the top-left corner of the smaller image
    y_pos = 200  # y-coordinate of the top-left corner of the smaller image

    # Copy the smaller image into the larger image using NumPy array slicing
    image[y_pos:y_pos+h, x_pos:x_pos+w] = crop_img

    resized_img = cv2.resize(image,(244,244))
    # Display the result
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    