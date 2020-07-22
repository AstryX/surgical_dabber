from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
#from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from pypcd import pypcd
import matplotlib.pyplot as plt
import umap
import cv2
import numpy as np
import math
from joblib import dump
import time
    
def readImagesAndMasks(path, num_of_images, is_num_singular):
    temp_im = []
    temp_mask = []
    temp_hsv_im = []
    num_loop = num_of_images
    if is_num_singular == True:
        num_loop = 1
    for i in range(num_loop):
        file_name = ""
        mask_name = ""
        if is_num_singular:
            file_name = path + "Img/img (" + str(num_of_images) + ").jpg"
            mask_name = path + "Mask/mask (" + str(num_of_images) + ").jpg"
            print('Reading img nr. ' + str(num_of_images))
        else:
            file_name = path + "Img/img (" + str(i+1) + ").jpg"
            mask_name = path + "Mask/mask (" + str(i+1) + ").jpg"
            print('Reading img nr. ' + str(i+1))
        img = cv2.imread(file_name)  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        temp_im.append(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        temp_hsv_im.append(img)
        mask = cv2.imread(mask_name)  
        temp_mask.append(mask)
    return np.array((temp_im)), np.array((temp_mask)), np.array((temp_hsv_im)) 
    
def extractColourFeatures(im_array, hsv_array, hsv_wrap_amount):
    #track_id = 0
    np_sum = np.sum(im_array, axis=3, keepdims=True)
    np_sum[np_sum == 0] = 1
    np_sum = np.repeat(np_sum, 3, axis=3)
    image_features = im_array.astype(float)
    image_features = np.divide(image_features, np_sum)
    image_features = np.concatenate((hsv_array, image_features), axis=3)
    
    '''for it in im_array:
        temp_image_features = []
        track_id += 1
        print('Extracting features for nr. ' + str(track_id))
        hsv_im = cv2.cvtColor(it, cv2.COLOR_RGB2HSV)
        for pixel_i in range(len(it)):
            line_im_features = []
            for pixel_j in range(len(it[pixel_i])):
                rgb_pixels = [it[pixel_i][pixel_j][0], it[pixel_i][pixel_j][1], 
                    it[pixel_i][pixel_j][2]]
                hsv_pixels = [((hsv_wrap_amount + hsv_im[pixel_i][pixel_j][0]) % 180), hsv_im[pixel_i][pixel_j][1], 
                    hsv_im[pixel_i][pixel_j][2]]
                pixel_sum = int(rgb_pixels[0]) + int(rgb_pixels[1]) + int(rgb_pixels[2])
                if pixel_sum == 0:
                    pixel_sum = 1
                cur_pixel_features = [hsv_pixels[0], hsv_pixels[1], hsv_pixels[2], 
                    rgb_pixels[0]/pixel_sum, rgb_pixels[1]/pixel_sum, rgb_pixels[2]/pixel_sum]
                print(cur_pixel_features)
                line_im_features.append(cur_pixel_features)
            temp_image_features.append(line_im_features)
        image_features.append(temp_image_features)'''
    return np.array((image_features))
    
def extractNeighbourFeatures(im_array, should_use_neighbours, should_exclude_thresholded,
    one_pixel_features, pixel_neighbourhood_size, num_pixel_features, hsv_v_index,
    image_size_rows, image_size_cols, neighbourhood_step, hsv_v_tolerance):
    image_features = []
    track_id = 0
    tolerance_removal_count = 0
    center_slot = 0
    if should_use_neighbours == True:
        center_slot = int(one_pixel_features * pixel_neighbourhood_size + one_pixel_features)
    inclusion_mask = np.ones((len(im_array))*image_size_rows*image_size_cols)
    track_pixel = -1
    for it in im_array:
        track_id += 1
        print('Extracting neighbour features for nr. ' + str(track_id))
        for pixel_i in range(len(it)):
            for pixel_j in range(len(it[pixel_i])):
                track_pixel += 1
                cur_pixel_features = []
                if should_use_neighbours == True:
                    cur_pixel_features = np.negative(np.ones(num_pixel_features))
                    cur_i_up = int(pixel_i - neighbourhood_step)
                    cur_i_down = int(pixel_i + neighbourhood_step)
                    cur_j_up = int(pixel_j - neighbourhood_step)
                    cur_j_down = int(pixel_j + neighbourhood_step)
                    if ((cur_i_up < 0) or (cur_i_down > (image_size_rows - 1))
                        or (cur_j_up < 0) or (cur_j_down > (image_size_cols - 1))):
                        image_features.append(cur_pixel_features) #Dummy features
                        inclusion_mask[track_pixel] = 0
                        continue
                    cur_pixel_features = (it[cur_i_up:cur_i_down+1,cur_j_up:cur_j_down+1]).flatten()
                else:
                    cur_pixel_features = it[pixel_i][pixel_j]
                    
                if cur_pixel_features[center_slot + hsv_v_index] >= hsv_v_tolerance:
                    image_features.append(cur_pixel_features)
                else:
                    tolerance_removal_count += 1
                    inclusion_mask[track_pixel] = 0
                    if should_exclude_thresholded == False:
                        image_features.append(cur_pixel_features)
                        
    print('HSV Value tolerance check removed ' + str(tolerance_removal_count) + ' pixels!')
    return np.array((image_features)),  np.array((inclusion_mask))
    
def extractMaskLabels(mask_images, image_features, inclusion_mask):
    mask_labels = []
    blood_data = []
    nonblood_data = []
    blood_labels = []
    nonblood_labels = []
    track_id = 0
    track_pixel = -1
    track_accepted_pixels = 0
    for it in mask_images:
        track_id += 1
        print('Retrieving masks for nr. ' + str(track_id))
        for pixel_i in range(len(it)):
            for pixel_j in range(len(it[pixel_i])):
                track_pixel += 1
                rgb_pixels = [it[pixel_i][pixel_j][0], it[pixel_i][pixel_j][1], 
                    it[pixel_i][pixel_j][2]]
                pixel_sum = int(rgb_pixels[0]) + int(rgb_pixels[1]) + int(rgb_pixels[2])
                if pixel_sum == 0:
                    if inclusion_mask[track_pixel] == 0:
                        continue
                    blood_data.append(image_features[track_accepted_pixels])
                    blood_labels.append(1)
                    mask_labels.append(1)
                    track_accepted_pixels += 1
                else:
                    if inclusion_mask[track_pixel] == 0:
                        continue
                    nonblood_data.append(image_features[track_accepted_pixels])
                    nonblood_labels.append(0)
                    mask_labels.append(0)
                    track_accepted_pixels += 1
    return [np.array((mask_labels)), np.array((blood_data)), np.array((nonblood_data)), 
        np.array((blood_labels)), np.array((nonblood_labels))]
        
def loadPointCloud(path):
    cur_cloud = pypcd.PointCloud.from_path(path)
    pcd_data = cur_cloud.pc_data
    return pcd_data
    
