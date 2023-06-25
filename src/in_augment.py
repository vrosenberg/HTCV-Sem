import cv2
import numpy as np
import json
import re
import os
from data_preprocessor import calc_average_damage
from shapely.geometry import Point, Polygon, box
import math
import random
import time


np.random.seed(int(time.time()))

# Constants

RESNET_INPUT_SHAPE = (224,224)
INITIAL_PATCH_SHAPE = (48,48)
RESIZE_PATCH_SHAPE = (134,134)
LABELS_PATH = "./data/train/labels/"

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
        points_str = polygon_str.replace('(', "").replace(')', "").replace("POLYGON ","").replace(',',"")
        points = [float(point) for point in points_str.split(' ')]

        for i in range(0,len(points),2):
            polygon_points.append((points[i],points[i+1]))
        polygon_points_list.append((buildID,polygon_points))
    
    return polygon_points_list

def scale_polygon_points(polygon_points, scale_factor):
    return[(x * scale_factor, y * scale_factor) for x, y in polygon_points]

def get_building_intersections(label_data, scale_factor, patch_polygon):
    scaled_polygon_points_list = [(buildID,scale_polygon_points(polygon_points, scale_factor)) for buildID, polygon_points in get_polygon_points_list(label_data)]
    building_polygon_list = [(buildID, Polygon(scaled_polygon_points)) for buildID, scaled_polygon_points in scaled_polygon_points_list]
    return [buildID for buildID, building_polygon in building_polygon_list if(patch_polygon.intersects(building_polygon))]
    
#def recalculate_damage_level(label_data)
def generate_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

# Generate a random color

'''
def get_random_patch(image, H_patch = 128, W_patch = 128):
    
    patch_size = [H_patch, W_patch]

    min_x = 0
    min_y = 0

    max_x = image.shape[1] - patch_size[1]
    max_y = image.shape[0] - patch_size[0]

    x = np.random.randint(min_x,max_x+1)
    y = np.random.randint(min_y,max_y+1)

    patch = image[y:y+patch_size[0],x:x+patch_size[1]]


    return (patch,{ "start_point" : (x,y),
                    "end_point" : (x+patch_size[1],y+patch_size[0])})
'''
def get_random_patches(image, amount_of_patches):
    patches = [get_random_patch(image) for i in range(amount_of_patches)]
        
    return

def resize_patch_coords(patch_coords, scale_x, scale_y):
    resized_patch_coords = {
        "start_point": patch_coords["start_point"],
        "end_point" : (patch_coords["start_point"][0] + (patch_coords["end_point"][0] - patch_coords["start_point"][0])*scale_x ,
                       patch_coords["start_point"][1] + (patch_coords["end_point"][1] - patch_coords["start_point"][1])*scale_y )                  
    }

    return resized_patch_coords

def resize_patch(patch,copied_patch_coords):
    resized_patch = cv2.resize(patch, RESIZE_PATCH_SHAPE)  
    patch_coords = resize_patch_coords(copied_patch_coords, resized_patch.shape[1]/patch.shape[1], resized_patch.shape[0]/patch.shape[0])
    return resized_patch, patch_coords

def scale_back_patch_coords(resized_patch_coords, patch_coords, scale_x,scale_y):
    scaled_patch_coords = {
        "start_point": (patch_coords["start_point"][0] + (resized_patch_coords["start_point"][0] - patch_coords["start_point"][0])*scale_x,
                        patch_coords["start_point"][1] + (resized_patch_coords["start_point"][1] - patch_coords["start_point"][1])*scale_y),
        "end_point" : (patch_coords["start_point"][0] + (resized_patch_coords["end_point"][0] - patch_coords["start_point"][0])*scale_x,
                        patch_coords["start_point"][1] + (resized_patch_coords["end_point"][1] - patch_coords["start_point"][1])*scale_y)
    }
    return scaled_patch_coords
if __name__ == "__main__":
    #img_name = "src/socal-fire_00001386_post_disaster.png"
    #json_name = "src/socal-fire_00001386_post_disaster.json"

    img_name = "src/guatemala-volcano_00000000_post_disaster.png"
    json_name = "src/guatemala-volcano_00000000_post_disaster.json"

    image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)

    resized_img = cv2.resize(image, RESNET_INPUT_SHAPE)

    resized_img = np.zeros((4,4))
    # Start loop here 
    patch, patch_coords = get_random_patch(resized_img,INITIAL_PATCH_SHAPE[0],INITIAL_PATCH_SHAPE[1])


    #patch_polygon = box(patch_coords["start_point"][0], patch_coords["start_point"][1], patch_coords["end_point"][0], patch_coords["end_point"][1])
    #print(copied_patch_coords["end_point"][0]-copied_patch_coords["start_point"][0])
    #print(copied_patch_coords["end_point"][1]-copied_patch_coords["start_point"][1])
    #print(copied_patch_coords)
    #cv2.imshow("patch",patch)
    #cv2.waitKey(0)
    #Assumes same scale factor for both axis
    #scale_factor = resized_img.shape[0] / image.shape[0]


    #building_intersections = get_building_intersections(label_data, scale_factor, patch_polygon)

    # Resize Patch
    resized_patch, resized_patch_coords = resize_patch(patch, patch_coords)
    
    #print(resized_patch.shape[0]/patch.shape[0])
    #print(resized_patch.shape[1]/patch.shape[1])

    #print(patch_coords)
    #print(resized_patch_coords)

    
#    cv2.imshow("patch", patch)
    #print(scale_patch_coords(pasted_patch_coords,patch.shape[1]/resized_patch.shape[1],patch.shape[0]/resized_patch.shape[0] ))
#    cv2.imshow("resized", resized_patch)
    #cv2.waitKey(0)
    # Paste Patch
    #resized_img = paste_patch_at_random_location(resized_img,resized_patch)
    

    
    #paste_patch_at_random_location(image,patch,patch_coords)
    aug_image, unscaled_copy_coords, paste_coords = paste_patch_at_random_location(resized_img, resized_patch, resized_patch_coords)
    copy_coords = scale_back_patch_coords(unscaled_copy_coords, patch_coords, patch.shape[1]/resized_patch.shape[1], patch.shape[0]/resized_patch.shape[0])

    paste_box = patch_polygon = box(paste_coords["start_point"][0], paste_coords["start_point"][1], paste_coords["end_point"][0], paste_coords["end_point"][1])
    copy_box = patch_polygon = box(copy_coords["start_point"][0], copy_coords["start_point"][1], copy_coords["end_point"][0], copy_coords["end_point"][1])

    # TODO go over buildings of image and recalc label, using polygons

    #copy_box
    #print(copy_coords)
    #print(paste_coords)
    cv2.imshow("test",aug_image)
    #cv2.waitKey(0)

    #print(aug_patch_coords)
    f = open(json_name)
    label_data = json.load(f)

    #print(patch)

    #print(image)

    #cv2.imshow("test", image)
    #cv2.waitKey(0)




    #print(building_intersections)



# Verification
#    polygons = label_data['features']['xy']
#    polygon_points_list = []
#    for buildID, building in enumerate(polygons):
#        polygon_str = building['wkt']
#        polygon_points = []
#        points_str = polygon_str.replace('(', "").replace(')', "").replace("POLYGON ","").replace(',',"")
#        points = [float(point) for point in points_str.split(' ')]
#
#        for i in range(0,len(points),2):
#            polygon_points.append((points[i],points[i+1]))
#        polygon_points_list.append((buildID,polygon_points))
#
#        min_x = math.floor(min(point[0] for point in polygon_points) * scale_factor)
#        max_x = math.ceil(max(point[0] for point in polygon_points) * scale_factor)
#
#        min_y = math.floor(min(point[1] for point in polygon_points) * scale_factor)
#        max_y = math.ceil(max(point[1] for point in polygon_points) * scale_factor)#
#
#        rectangle_params= (min_x, min_y, max_x-min_x, max_y-min_y)
#        random_color = generate_random_color()
#        
#        cv2.rectangle(resized_img,rectangle_params , random_color, -1)


    
    
    #intersections = 
    #json_path = "./data/" + "train" + "/labels/" + json_name



    # From Here everything is in scaled space (RESNET input size)

    #min_x = math.floor(min(point[0] for point in polygon_points) * scale_factor)
    #max_x = math.ceil(max(point[0] for point in polygon_points) * scale_factor)

    #min_y = math.floor(min(point[1] for point in polygon_points) * scale_factor)
    #max_y = math.ceil(max(point[1] for point in polygon_points) * scale_factor)
    
    #print(scaled_polygon_points_list)
    #rectangle_params= (min_x, min_y, max_x-min_x, max_y-min_y)
    
    #patch_params = (patch_coords["start_point"][0], patch_coords["start_point"][1], patch_coords["end_point"][0] - patch_coords["start_point"][0], patch_coords["end_point"][1] - patch_coords["start_point"][1])
    
    #translation = 0
    #test_params = (min_x -translation, min_y -translation, max_x-min_x, max_y-min_y)
    
    #print(patch_coords)

    #print(building_intersections)

    #patch_polygon = box(patch_coords)

    
    #cv2.rectangle(resized_img,rectangle_params , (0, 0, 0), -1)
    #cv2.rectangle(resized_img, patch_params,(0, 0, 255), -1 )
    #cv2.rectangle(resized_img, test_params,(255, 0, 0), -1 )

    
    
    #test_polygon = box(min_x-translation, min_y-translation, max_x-translation, max_y-translation)
    #if(patch_polygon.intersects(building_polygon)):
    #    print("INTERSECTION")
    #if(patch_polygon.contains(building_polygon)):
    #    print("CONTAINED")


    # Resize back to original
    #cv2.imshow("test",cv2.resize(resized_img,image.shape[:2]))
    #cv2.imwrite("10buildings.png",cv2.resize(resized_img,image.shape[:2]))
    #cv2.waitKey(0)
    
    

    '''
    polygons = label_data['features']['xy']
    for building in polygons:
        polygon_str = building['wkt']
        polygon_points = []
        points_str = polygon_str.replace('(', "").replace(')', "").replace("POLYGON ","").replace(',',"")
        points = [float(point) for point in points_str.split(' ')]

        for i in range(0,len(points),2):
            polygon_points.append((points[i],points[i+1]))
        Polygon(polygon_points)
    
    '''

    #copy
    #patch = get_random_patch(image)

    #cv2.imshow("test",patch)
    #cv2.waitKey(0)
    #get_copied_patch(path)
    #image = np.zeros((5,5))
    
    #take random patch+
    #(x,y)
    #patch = get_random_patch(image) 
    '''
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

    json_path = "./data/" + data_type + "/labels/" + name.split('.')[0]+".json"

    patches = get_building_patches(json_path=json_path)

    path = 'data/train/images/'+name
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    for (y,h,x,w,label) in patches:
        mask = np.zeros([h,w,image.shape[2]], dtype=np.uint8)
        image[y:y+h,x:x+w] = mask
    ci'''

    #'POLYGON ((917.7720847921884 732.7062537792558, 907.6213628757642 725.1242804382573, 914.9839923437013 715.9347330488779, 925.5281487485797 722.1464920387161, 922.3015423879609 727.7179381319243, 917.7720847921884 732.7062537792558))'


    '''
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

    '''
    '''
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
    