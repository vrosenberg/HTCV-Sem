import cv2
import numpy as np
import json
import re
import os
from shapely.geometry import Point, Polygon, box
import math
import random
import time
from PIL import Image, ImageDraw


np.random.seed(int(time.time()))

# Constants
DAMAGE_LEVEL_TO_SCORE = {
    "destroyed" : 4,
    "major-damage" : 3,
    "minor-damage" : 2,
    "no-damage" : 1,
    "no-building" : 0
}

BASE_DATA_PATH = "only_all/"
DS_SIZE = 2799
AMOUNT_OF_SAMPLES = 28000

RESNET_INPUT_SHAPE = (224,224)
INITIAL_PATCH_SHAPE = (48,48)
RESIZE_PATCH_SHAPES = [(134,134),(80,80),(48,48)]
IMAGE_SHAPE = (1024,1024)
LABELS_PATH = "./data/train/labels/"
IMAGES_PATH = "./data/train/images/"
AUGMENTED_DATA_PATH = "./augmented_data/"


def get_random_patch(image, H_patch, W_patch):
    
    patch_size = [H_patch, W_patch]

    min_x = 1 - patch_size[1] 
    min_y = 1 - patch_size[0]

    max_x = image.shape[0] - 1 
    max_y = image.shape[1] - 1

    x = np.random.randint(min_x,max_x+1)
    y = np.random.randint(min_y,max_y+1)
    x_pos = x
    y_pos = y
    #trim patch_size according to random crop pos
    actual_patch_size = [patch_size[0],patch_size[1]]
    if(x < 0):
        actual_patch_size[1] = x + patch_size[1]
        x_pos = 0
    if(y < 0):
        actual_patch_size[0] = y + patch_size[0]
        y_pos = 0
    if(x + patch_size[1] > image.shape[0]):
        actual_patch_size[1] = image.shape[0] - x
    if(y + patch_size[0] > image.shape[1]):
        actual_patch_size[0] = image.shape[1] - y

    patch = image[y_pos:y_pos+actual_patch_size[0],x_pos:x_pos+actual_patch_size[1]]

    return (patch,{ "start_point" : (x_pos,y_pos),
                    "end_point" : (x_pos+actual_patch_size[1],y_pos+actual_patch_size[0])})

def paste_patch_at_random_location(image,patch, patch_coords):
    min_x = 1 - patch.shape[1] 
    min_y = 1 - patch.shape[0]

    max_x = image.shape[0] - 1 
    max_y = image.shape[1] - 1

    x = np.random.randint(min_x,max_x+1)
    y = np.random.randint(min_y,max_y+1)
    x_pos = x
    y_pos = y

    x_patch_pos = 0
    y_patch_pos = 0
    #trim patch_size according to random crop pos
    actual_patch_size = [patch.shape[0],patch.shape[1]]
    if(x < 0):
        actual_patch_size[1] = x + patch.shape[1]
        x_pos = 0
        x_patch_pos = x_patch_pos - x
    if(y < 0):
        actual_patch_size[0] = y + patch.shape[0]
        y_pos = 0
        y_patch_pos = y_patch_pos - y
    if(x + patch.shape[1] > image.shape[0]):
        actual_patch_size[1] = image.shape[0] - x
    if(y + patch.shape[0] > image.shape[1]):
        actual_patch_size[0] = image.shape[1] - y

    clipped_patch = patch[y_patch_pos:y_patch_pos+actual_patch_size[0],x_patch_pos:x_patch_pos+actual_patch_size[1]]
    image[y_pos:y_pos+actual_patch_size[0],x_pos:x_pos+actual_patch_size[1]] = clipped_patch
    
    copy_coords = { 
        "start_point" : (patch_coords["start_point"][0]+ x_patch_pos, patch_coords["start_point"][1] + y_patch_pos),
        "end_point" : (patch_coords["start_point"][0] + x_patch_pos+actual_patch_size[1], patch_coords["start_point"][1] + y_patch_pos+actual_patch_size[0])
    }

    paste_coords = {
        "start_point" : (x_pos,y_pos),
        "end_point" : (x_pos+actual_patch_size[1],y_pos+actual_patch_size[0])
    }
    return image, copy_coords, paste_coords

def get_polygon_points_list(json_data):
    polygons = json_data['features']['xy']
    polygon_points_list = []

    for buildID, building in enumerate(polygons):
        polygon_str = building['wkt']
        polygon_points = []
        dmg_score = -1
        points_str = polygon_str.replace('(', "").replace(')', "").replace("POLYGON ","").replace(',',"")
        points = [float(point) for point in points_str.split(' ')]
        
        for i in range(0,len(points),2):
            polygon_points.append((points[i],points[i+1]))

        if(building["properties"]["feature_type"] == 'building' and building["properties"]["subtype"] != 'un-classified'):
            dmg_score = DAMAGE_LEVEL_TO_SCORE[building["properties"]["subtype"]]

        polygon_points_list.append((dmg_score,polygon_points))
        
    return polygon_points_list

def scale_polygon_points(polygon_points, scale_factor):
    return[(x * scale_factor, y * scale_factor) for x, y in polygon_points]

def get_building_intersections(label_data, scale_factor, patch_polygon):
    scaled_polygon_points_list = [(buildID,scale_polygon_points(polygon_points, scale_factor)) for buildID, polygon_points in get_polygon_points_list(label_data)]
    building_polygon_list = [(buildID, Polygon(scaled_polygon_points)) for buildID, scaled_polygon_points in scaled_polygon_points_list]
    return [buildID for buildID, building_polygon in building_polygon_list if(patch_polygon.intersects(building_polygon))]
    
def paste_patch(image,patch, target, target_patch):
    min_x = 1 - patch.shape[1] 
    min_y = 1 - patch.shape[0]

    max_x = image.shape[0] - 1 
    max_y = image.shape[1] - 1

    x = np.random.randint(min_x,max_x+1)
    y = np.random.randint(min_y,max_y+1)
    x_pos = x
    y_pos = y

    x_patch_pos = 0
    y_patch_pos = 0
    #trim patch_size according to random crop pos
    actual_patch_size = [patch.shape[0],patch.shape[1]]
    if(x < 0):
        actual_patch_size[1] = x + patch.shape[1]
        x_pos = 0
        x_patch_pos = x_patch_pos - x
    if(y < 0):
        actual_patch_size[0] = y + patch.shape[0]
        y_pos = 0
        y_patch_pos = y_patch_pos - y
    if(x + patch.shape[1] > image.shape[0]):
        actual_patch_size[1] = image.shape[0] - x
    if(y + patch.shape[0] > image.shape[1]):
        actual_patch_size[0] = image.shape[1] - y

    clipped_patch = patch[y_patch_pos:y_patch_pos+actual_patch_size[0],x_patch_pos:x_patch_pos+actual_patch_size[1]]
    image[y_pos:y_pos+actual_patch_size[0],x_pos:x_pos+actual_patch_size[1]] = clipped_patch
    
    clipped_target_patch = target_patch[y_patch_pos:y_patch_pos+actual_patch_size[0],x_patch_pos:x_patch_pos+actual_patch_size[1]]
    target[y_pos:y_pos+actual_patch_size[0],x_pos:x_pos+actual_patch_size[1]] = clipped_target_patch
    
    return image.copy(), target.copy()

def copy_patch(image, target, H_patch, W_patch, patch_number):
    
    patch_size = [H_patch, W_patch]

    min_x = 1 - patch_size[1] 
    min_y = 1 - patch_size[0]

    max_x = image.shape[0] - 1 
    max_y = image.shape[1] - 1

    x = np.random.randint(min_x,max_x+1)
    y = np.random.randint(min_y,max_y+1)
    x_pos = x
    y_pos = y
    #trim patch_size according to random crop pos
    actual_patch_size = [patch_size[0],patch_size[1]]
    if(x < 0):
        actual_patch_size[1] = x + patch_size[1]
        x_pos = 0
    if(y < 0):
        actual_patch_size[0] = y + patch_size[0]
        y_pos = 0
    if(x + patch_size[1] > image.shape[0]):
        actual_patch_size[1] = image.shape[0] - x
    if(y + patch_size[0] > image.shape[1]):
        actual_patch_size[0] = image.shape[1] - y

    image_patch = image[y_pos:y_pos+actual_patch_size[0],x_pos:x_pos+actual_patch_size[1]]
    target_patch =  target[y_pos:y_pos+actual_patch_size[0],x_pos:x_pos+actual_patch_size[1]]

    return image_patch.copy(), update_target_patch(target_patch, patch_number)

def update_target_patch(target_patch, patch_number):
    
    updated_patch = target_patch.copy()
    for y in range(target_patch.shape[0]):
        for x in range(target_patch.shape[1]):
            pixel_val = target_patch[y,x]
            target_pixel = pixel_val.copy() 
            target_pixel[1] = patch_number
            updated_patch[y,x] = target_pixel
    return updated_patch


def get_pixel_data(label_data, shape, scale_factor):
    polygon_points_list = get_polygon_points_list(label_data)
    
    ref_image = np.zeros((shape[0],shape[1],3), dtype=np.int32) - 1
    for idx, (dmgLevel, polygon_points) in enumerate(polygon_points_list):
        scaled_polygon_points = [(polygon_point[0] * scale_factor , polygon_point[1] * scale_factor) for polygon_point in polygon_points]
        scaled_polygon_points = np.round(scaled_polygon_points).astype(np.int32)
        cv2.fillPoly(ref_image, [scaled_polygon_points], color=(idx,0,dmgLevel))
    return ref_image


def get_memory_map(ref_data):
    memory_map = {}

    for y in range(ref_data.shape[0]):
        for x in range(ref_data.shape[1]):
            pixel_val = ref_data[y,x]
            id = pixel_val[0]
            copy_patch_number = pixel_val[1]
            dmg_level = pixel_val[2]
            
            if(dmg_level < 0):
                continue
            if id not in memory_map:
                memory_map[id] = {}
                if copy_patch_number not in memory_map[id]:
                    memory_map[id][copy_patch_number] = dmg_level
            elif copy_patch_number not in memory_map[id]:
                    memory_map[id][copy_patch_number] = dmg_level
    return memory_map

def calc_average_damage(memory_map):
    total = 0.0
    count = 0.0

    for key in memory_map:
        for sub_key in memory_map[key]:
            if memory_map[key][sub_key] is not None:
                total += memory_map[key][sub_key]
                count += 1

    if(count == 0):
        average = 0
    else:
        average = round(total/count)

    return average
if __name__ == "__main__":

    json_files = os.listdir(LABELS_PATH)

    class_instances_orig = [0,0,0,0,0]
    class_instances = [0,0,0,0,0]

    #for i in range(AMOUNT_OF_SAMPLES):

    with open("data/xview2.txt","r") as file:
        xview_entries = file.readlines()
    xview_entries = [line.strip() for line in xview_entries]
    xview_entries = xview_entries[:DS_SIZE]

    for i, file in enumerate(xview_entries):
        #file = np.random.choice(xview_entries)
        if "post" not in file:
            continue
        print(file)
        img_name = file.split('.')[0] + '.png'
        json_name = file.split('.')[0] + '.json'
        image = cv2.imread(IMAGES_PATH + img_name, cv2.IMREAD_UNCHANGED)
        scale_factor = RESNET_INPUT_SHAPE[0]/image.shape[0]
        
        resized_img = cv2.resize(image, RESNET_INPUT_SHAPE)
        augmented_image = resized_img.copy()
        
        f = open(LABELS_PATH+json_name)
        label_data = json.load(f)

        target_data = get_pixel_data(label_data, RESNET_INPUT_SHAPE, scale_factor)
        augmented_target = target_data.copy()

        memory_map = get_memory_map(augmented_target)
        orig_avg = calc_average_damage(memory_map)
        #print("original average: " + str(orig_avg))
        for idx, resize_shape in enumerate(RESIZE_PATCH_SHAPES):
            
            # Copy from original image/ref
            patch, target_patch = copy_patch(resized_img, target_data, INITIAL_PATCH_SHAPE[0],INITIAL_PATCH_SHAPE[1], idx + 1)

            # Resize it
            resized_patch = cv2.resize(patch, resize_shape)
            resized_target_patch = cv2.resize(target_patch, resize_shape, interpolation=cv2.INTER_NEAREST)

            # Paste it
            augmented_image, augmented_target = paste_patch(augmented_image, resized_patch, augmented_target, resized_target_patch)
            
        memory_map = get_memory_map(augmented_target)
        avg = calc_average_damage(memory_map)

        #print("new average: " + str(avg))
        print(i)
        
        class_instances_orig[orig_avg] += 1
        class_instances[avg] += 1

        
        cv2.imwrite(AUGMENTED_DATA_PATH+ BASE_DATA_PATH + str(avg) + "/" + file.split('.')[0] +"_id_" + str(i) +".png", augmented_image)

        SAVE_VISUALIZED_EXAMPLES = False
        if SAVE_VISUALIZED_EXAMPLES:
            vals = []

            for y in range(augmented_target.shape[0]):
                for x in range(augmented_target.shape[1]):
                    pixel_val = augmented_target[y,x]
                    if(pixel_val[2] >= 0):
                        vals.append((y,x,pixel_val[1]))
                        
            for y,x, patch_number in vals:
                p = augmented_image[y,x]

                #blue
                if(patch_number == 0):
                    p[0] = 255
                    p[1] = 0
                    p[2] = 0
                
                #gree
                if(patch_number == 1):
                    p[0] = 0
                    p[1] = 255
                    p[2] = 0
                #red
                if(patch_number == 2):
                    p[0] = 0
                    p[1] = 0
                    p[2] = 255
                #cyan
                if(patch_number == 3):
                    p[0] = 255
                    p[1] = 233
                    p[2] = 0
                augmented_image[y,x] = p
            

            vals = []
            for y in range(target_data.shape[0]):
                for x in range(target_data.shape[1]):
                    pixel_val = target_data[y,x]
                    if(pixel_val[2] >= 0):
                        vals.append((y,x))
                        
            for y,x in vals:
                #blue
                p = resized_img[y,x]
                p[0] = 255
                p[1] = 0
                p[2] = 0
                resized_img[y,x] = p

            if( orig_avg == avg):
                cv2.imwrite("augmented_data/visual_examples/" + file.split('.')[0] + "_labeled.png", resized_img)
                cv2.imwrite("augmented_data/visual_examples/" + file.split('.')[0] + "_augmented.png", augmented_image)
            else:
                cv2.imwrite("augmented_data/avg_diff_examples/" + file.split('.')[0] + "_"+ str(orig_avg) +"_labeled.png", resized_img)
                cv2.imwrite("augmented_data/avg_diff_examples/" + file.split('.')[0] + "_"+ str(avg) + "_augmented.png", augmented_image)
    
    print("class instances:")
    print(class_instances)
    print("")
    print("class_instances_orig:")
    print(class_instances_orig)
