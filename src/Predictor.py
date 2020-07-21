from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import math
from joblib import load
import json
import time
from statistics import mean
from sklearn.metrics import confusion_matrix
from Helper import readImagesAndMasks, extractColourFeatures
from Helper import extractNeighbourFeatures, extractMaskLabels
from Helper import loadPointCloud

def computeAndDisplayContours(print_text, passed_mask, draw_image, remove_small, bool_display_all_contours, pool_area_threshold):
    final_contours = None
    temp_image = np.copy(draw_image)
    gray_mask = cv2.cvtColor(passed_mask, cv2.COLOR_BGR2GRAY)
    print(print_text)
    contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    final_contours = contours
    cv2.drawContours(temp_image, contours, -1, (0,0,255), cv2.FILLED)
    dual_image = np.concatenate((draw_image, temp_image), axis=1)
    predicted_final = np.copy(draw_image)
    predicted_final[:,:,0] = 0
    predicted_final[:,:,1] = 0
    predicted_final[:,:,2] = 0
    cv2.drawContours(predicted_final, contours, -1, (0,0,255), cv2.FILLED)
    
    if remove_small == True:
        new_contours = []
        temp_image_pool = np.copy(draw_image)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area >= pool_area_threshold:
                new_contours.append(contours[i])
        cv2.drawContours(temp_image_pool, new_contours, -1, (0,0,255), cv2.FILLED)
        dual_image = np.concatenate((dual_image, temp_image_pool), axis=1)
        
        predicted_final = np.copy(draw_image)
        predicted_final[:,:,0] = 0
        predicted_final[:,:,1] = 0
        predicted_final[:,:,2] = 0
        cv2.drawContours(predicted_final, new_contours, -1, (0,0,255), cv2.FILLED)
        final_contours = new_contours

    if bool_display_all_contours == True:
        cv2.imshow(print_text, dual_image) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return predicted_final, final_contours

def displayNormalization(pred_features, image_size_rows, image_size_cols, full_image):
    print('Checking normalization effect')
    print(pred_features.shape)
    disp_features_r = ((pred_features[:,3] - min(pred_features[:,3])) / 
        (max(pred_features[:,3]) - min(pred_features[:,3])))
    disp_features_g = ((pred_features[:,4] - min(pred_features[:,4])) / 
        (max(pred_features[:,4]) - min(pred_features[:,4])))
    disp_features_b = ((pred_features[:,5] - min(pred_features[:,5])) / 
        (max(pred_features[:,5]) - min(pred_features[:,5])))
    disp_combined = []
    for i in range(image_size_rows):
        interm_arr = []
        for j in range(image_size_cols):
            cur_idx = int(i*image_size_cols + j)
            interm_arr.append([disp_features_b[cur_idx], disp_features_g[cur_idx], disp_features_r[cur_idx]])
        disp_combined.append(interm_arr)
    disp_combined = np.array((disp_combined))
    #disp_combined = disp_combined * 255
    print('Mins')
    print((disp_combined[:,:,0].flatten()).shape)
    print(min(disp_combined[:,:,0].flatten()))
    print(min(disp_combined[:,:,1].flatten()))
    print(min(disp_combined[:,:,2].flatten()))
    print('Maxs')
    print(max(disp_combined[:,:,0].flatten()))
    print(max(disp_combined[:,:,1].flatten()))
    print(max(disp_combined[:,:,2].flatten()))
    print('Means')
    print(mean(disp_combined[:,:,0].flatten()))
    print(mean(disp_combined[:,:,1].flatten()))
    print(mean(disp_combined[:,:,2].flatten()))
    print(disp_combined.shape)
    temp_original = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
    disp_combined = np.concatenate((temp_original/255, disp_combined), axis=1)
    cv2.imshow('Standardized blood image', disp_combined) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()   

def compressPredictedNeighbourhoods(pred_features, image_size_rows, image_size_cols):
    mod_pred_features = []
    num_compressed_cols = int(math.floor(image_size_cols / 3))
    if ((image_size_cols % 3) == 2):
        num_compressed_cols += 1
        
    num_compressed_rows = int(math.floor(image_size_rows / 3))
    if ((image_size_rows % 3) == 2):
        num_compressed_rows += 1
        
    for i in range(num_compressed_rows):
        for j in range(num_compressed_cols):
            cur_idx = int((image_size_cols + 1) + 3*j + 3*image_size_cols*i)
            mod_pred_features.append(pred_features[cur_idx])
            
    return mod_pred_features, num_compressed_cols, num_compressed_rows
        

def decompressPredictedNeighbourhoods(pred_labels, image_size_rows, image_size_cols, compressed_cols, compressed_rows): 
    reconstructed_labels = np.ones((image_size_rows*image_size_cols), np.uint8)
    labels_idx = 0
    for i in range(compressed_rows):
        for j in range(compressed_cols):
            if pred_labels[labels_idx] == 1:
                cur_idx = int((image_size_cols + 1) + 3*j + 3*image_size_cols*i)
                #Center
                reconstructed_labels[cur_idx] = 1
                remainder = int(cur_idx % image_size_cols)
                divider = int(math.floor(cur_idx / image_size_cols))
                
                if divider > 0:
                    #Up
                    reconstructed_labels[cur_idx-image_size_cols] = 1
                    
                if (divider < (image_size_rows-1)):
                    #Down
                    reconstructed_labels[cur_idx+image_size_cols] = 1
                
                if remainder > 0:
                    #Left
                    reconstructed_labels[cur_idx-1] = 1
                    
                    if divider > 0:
                        #Up Left
                        reconstructed_labels[cur_idx-image_size_cols-1] = 1
                        
                    if (divider < (image_size_rows-1)):
                        #Down Left
                        reconstructed_labels[cur_idx+image_size_cols-1] = 1
                
                if (remainder < (image_size_cols-1)):
                    #Right
                    reconstructed_labels[cur_idx+1] = 1
                    
                    if divider > 0:
                        #Up Right
                        reconstructed_labels[cur_idx-image_size_cols+1] = 1
                        
                    if (divider < (image_size_rows-1)):
                        #Down Right
                        reconstructed_labels[cur_idx+image_size_cols+1] = 1
            labels_idx += 1 
    return reconstructed_labels
    
def predictImageLabels(params_path, pred_image, im_path, base_path):
    #Default parameter values
    image_size_rows = 250
    image_size_cols = 330
    num_pixel_features = 54
    one_pixel_features = 6
    dim_red_features = 3
    dim_red_features_neighbour = 18
    hsv_v_tolerance = 30
    hsv_v_index = 2
    pool_area_threshold = 250
    hsv_wrap_amount = 90

    bool_add_neighbourhoods = True
    bool_should_mask_hue = True
    bool_should_normalize = False
    bool_should_dimensionally_reduce = True
    bool_exclude_border_pixels = True
    bool_do_normalization_display = False
    bool_display_all_contours = True
    bool_remove_small_pools = True
    bool_compress_predicted_pixels = False

    pixel_neighbourhood_size = 3
    neighbourhood_step = math.floor(pixel_neighbourhood_size / 2)
    im_path = im_path
    predict_num = pred_image
    classifier_name = base_path+"blood_classifier.joblib"
    pca_name = base_path+"pca.joblib"
    scaler_name = base_path+"scaler.joblib"
    #~~~~~~~~~~~~~ Default values finish

    with open(params_path) as json_data_file:
            
        data = json.load(json_data_file)
        
        if 'image_size_rows' in data:
            image_size_rows = data['image_size_rows']
        if 'image_size_cols' in data:
            image_size_cols = data['image_size_cols']
        if 'num_pixel_features' in data:
            num_pixel_features = data['num_pixel_features']
        if 'one_pixel_features' in data:
            one_pixel_features = data['one_pixel_features']
        if 'dim_red_features' in data:
            dim_red_features = data['dim_red_features']
        if 'dim_red_features_neighbour' in data:
            dim_red_features_neighbour = data['dim_red_features_neighbour']
        if 'hsv_v_tolerance' in data:
            hsv_v_tolerance = data['hsv_v_tolerance']
        if 'hsv_v_index' in data:
            hsv_v_index = data['hsv_v_index']
        if 'hsv_wrap_amount' in data:
            hsv_wrap_amount = data['hsv_wrap_amount']
        if 'bool_add_neighbourhoods' in data:
            bool_add_neighbourhoods = data['bool_add_neighbourhoods']
        if 'bool_should_mask_hue' in data:
            bool_should_mask_hue = data['bool_should_mask_hue']
        if 'bool_should_normalize' in data:
            bool_should_normalize = data['bool_should_normalize']
        if 'bool_should_dimensionally_reduce' in data:
            bool_should_dimensionally_reduce = data['bool_should_dimensionally_reduce']
        if 'bool_exclude_border_pixels' in data:
            bool_exclude_border_pixels = data['bool_exclude_border_pixels']
        if 'bool_do_normalization_display' in data:
            bool_do_normalization_display = data['bool_do_normalization_display']
        if 'bool_display_all_contours' in data:
            bool_display_all_contours = data['bool_display_all_contours']
        if 'bool_remove_small_pools' in data:
            bool_remove_small_pools = data['bool_remove_small_pools']
        if 'bool_compress_predicted_pixels' in data:
            bool_compress_predicted_pixels = data['bool_compress_predicted_pixels']
        if 'pixel_neighbourhood_size' in data:
            pixel_neighbourhood_size = data['pixel_neighbourhood_size']
            neighbourhood_step = math.floor(pixel_neighbourhood_size / 2)
        if 'classifier_name' in data:
            classifier_name = base_path+data['classifier_name']
        if 'pca_name' in data:
            pca_name = base_path+data['pca_name']
        if 'scaler_name' in data:
            scaler_name = base_path+data['scaler_name']
    
    if bool_add_neighbourhoods == True:
        dim_red_features = dim_red_features_neighbour   

    time_preprocessing = time.time()
    time_preprocessing_total = time_preprocessing
    full_image, mask_image = readImagesAndMasks(im_path, predict_num, True)
    print('Image Reading Time Taken:' + str(time.time()-time_preprocessing) + ' seconds.')
    time_preprocessing = time.time()
    singular_features = extractColourFeatures(full_image, hsv_wrap_amount)
    print('Single Feature Extraction Time Taken:' + str(time.time()-time_preprocessing) + ' seconds.')
    time_preprocessing = time.time()
    full_image = full_image[0]
    pred_features, inclusion_mask = extractNeighbourFeatures(singular_features, bool_add_neighbourhoods, False,
        one_pixel_features, pixel_neighbourhood_size, num_pixel_features, hsv_v_index,
        image_size_rows, image_size_cols, neighbourhood_step, hsv_v_tolerance, bool_exclude_border_pixels)
    print('Neighbourhood Extraction Time Taken:' + str(time.time()-time_preprocessing) + ' seconds.')
    time_preprocessing = time.time()
    dummy_inclusion_mask = np.ones(len(inclusion_mask))
    mask_labels, _, _, _, _ = extractMaskLabels(mask_image, pred_features, dummy_inclusion_mask)
    mask_image = mask_image[0]
    print('Mask Extraction Time Taken:' + str(time.time()-time_preprocessing) + ' seconds.')
    print('Total Preprocessing Time Taken:' + str(time.time()-time_preprocessing_total) + ' seconds.')


    clf = load(classifier_name) 
    dim_red_model = load(pca_name)
    scaler = load(scaler_name)

    print('Predicting pixel labels for the selected image...')
    if bool_should_normalize == True:
        pred_features = scaler.transform(pred_features)
    if bool_should_dimensionally_reduce == True:
        pred_features = dim_red_model.transform(pred_features)
        
    pred_labels = []
    time_prediction = time.time()
    if bool_compress_predicted_pixels == True:
        time_compression = time.time()
        compressed_features, compressed_cols, compressed_rows = compressPredictedNeighbourhoods(pred_features, image_size_rows, image_size_cols)
        pred_labels = clf.predict(compressed_features)
        pred_labels = decompressPredictedNeighbourhoods(pred_labels, image_size_rows, image_size_cols, compressed_cols, compressed_rows)
        print('Total Compression Time Taken:' + str(time.time()-time_compression) + ' seconds.')
    else:
        pred_labels = clf.predict(pred_features)
    print('Total Prediction Time Taken:' + str(time.time()-time_prediction) + ' seconds.')
        
    if ((bool_should_normalize == True) and (bool_do_normalization_display == True)):
        displayNormalization(pred_features, image_size_rows, image_size_cols, full_image)

    #Remove image parts that are too dark
    if bool_should_mask_hue == True:
        value_culling = 0
        for i in range(len(inclusion_mask)):
            if ((inclusion_mask[i] == 0) and (pred_labels[i] == 1)):
                pred_labels[i] = 0
                value_culling += 1
        print('Culled ' + str(value_culling) + ' dark pixel values that were classified as blood!')
    
    smoothing_kernel = np.ones((5, 5), np.uint8)
    pred_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
    inverted_mask = cv2.bitwise_not(mask_image)
    
    if bool_display_all_contours == True:
        _, _ = computeAndDisplayContours('Finding contours of the ground-truth blood labels', 
            inverted_mask, pred_image, bool_remove_small_pools, bool_display_all_contours, pool_area_threshold)

    '''print('Closing -> Opening ground-truth pixels')
    #Ground truth morphology
    closing = cv2.morphologyEx(inverted_mask, cv2.MORPH_CLOSE, smoothing_kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, smoothing_kernel)

    computeAndDisplayContours('Finding contours for morphologically changed ground-truth pixels', opening, pred_image, bool_remove_small_pools, pool_area_threshold)
    '''

    predicted_mask = np.copy(inverted_mask)
    predicted_mask[:,:,0] = 0
    predicted_mask[:,:,1] = 0
    predicted_mask[:,:,2] = 0

    for i in range(len(pred_labels)):
        if pred_labels[i] == 1:
            trunc_x = math.trunc((i)/image_size_cols)
            trunc_y = i - trunc_x * image_size_cols
            predicted_mask[trunc_x,trunc_y,0] = 255
            predicted_mask[trunc_x,trunc_y,1] = 255
            predicted_mask[trunc_x,trunc_y,2] = 255
    
    if bool_display_all_contours == True:
        _, _ = computeAndDisplayContours('Finding contours for initial predicted pixels',
            predicted_mask, pred_image, bool_remove_small_pools, bool_display_all_contours, pool_area_threshold)

    #Predicted morphology
    print('Closing -> Opening predicted pixels')
    closing_pred = cv2.morphologyEx(predicted_mask, cv2.MORPH_CLOSE, smoothing_kernel)
    opening_pred = cv2.morphologyEx(closing_pred, cv2.MORPH_OPEN, smoothing_kernel)

    morph_pred_labels = []
    for pixel_i in range(len(opening_pred)):
        for pixel_j in range(len(opening_pred[pixel_i])):
            rgb_pixels = [opening_pred[pixel_i][pixel_j][0], opening_pred[pixel_i][pixel_j][1], 
                opening_pred[pixel_i][pixel_j][2]]
            pixel_sum = int(rgb_pixels[0]) + int(rgb_pixels[1]) + int(rgb_pixels[2])
            if pixel_sum == 765:
                morph_pred_labels.append(1)
            else:
                morph_pred_labels.append(0)
                 
    final_y_pred_image, final_contours = computeAndDisplayContours('Finding contours for morphologically changed predicted pixels', 
        opening_pred, pred_image, bool_remove_small_pools, bool_display_all_contours, pool_area_threshold)

    final_y_pred_labels = np.zeros(len(mask_labels))
    track_pos = -1
    for i in range(len(final_y_pred_image)):
        for j in range(len(final_y_pred_image[i])):
            track_pos += 1
            if ((final_y_pred_image[i][j][0] == 0)and(final_y_pred_image[i][j][1] == 0)
                and(final_y_pred_image[i][j][2] == 255)):
                final_y_pred_labels[track_pos] = 1

    print('Confusion matrix of the prediction vs ground-truth:')
    print(confusion_matrix(mask_labels, final_y_pred_labels))
    
    return final_y_pred_labels, final_contours, pred_image

def findOptimalDestination(before_z, after_z, image_size_rows, image_size_cols, final_y_pred_labels, final_contours, pred_image, bool_display_final_contour):
    print('Beginning point cloud experimentation...')
    print('Final contours shape')
    print((np.array((final_contours))).shape)

    top_pool_id = 0
    top_pool_volume = 0
    top_pool_deepest_point_id = 0

    for k in range(len(final_contours)):
        contours_temp = np.copy(pred_image)
        contours_temp[:,:,0] = 0
        contours_temp[:,:,1] = 0
        contours_temp[:,:,2] = 0
        cv2.drawContours(contours_temp, np.array(([final_contours[k]])), -1, (0,0,255), cv2.FILLED)
        contours_pred_labels = np.zeros(len(final_y_pred_labels))
        
        track_pos = -1
        for i in range(len(contours_temp)):
            for j in range(len(contours_temp[i])):
                track_pos += 1
                if ((contours_temp[i][j][0] == 0)and(contours_temp[i][j][1] == 0)
                    and(contours_temp[i][j][2] == 255)):
                    contours_pred_labels[track_pos] = 1
            
        contour_volume = 0
        temp_deepest_point = 0
        temp_deepest_depth = 0
        for j in range(len(contours_pred_labels)):
            if contours_pred_labels[j] == 1:
                if (before_z[j] - after_z[j]) > 0:
                    if before_z[j] > temp_deepest_depth:
                        temp_deepest_depth = before_z[j]
                        temp_deepest_point = j
                    contour_volume = contour_volume + before_z[j] - after_z[j]
        print('Contour nr. ' + str(k+1) + ' volume is ' + str(contour_volume))
        print('Deepest z is ' + str(temp_deepest_depth))
        if top_pool_volume < contour_volume:
            top_pool_volume = contour_volume
            top_pool_id = k
            top_pool_deepest_point_id = temp_deepest_point
            
    print('Most volume contour id: ' + str(top_pool_id))
    print('Most volume contour volume: ' + str(top_pool_volume))
    print('Most volume contour deepest point ' + str(top_pool_deepest_point_id))
    print('Drawing the deepest pool...')

    temp_image = np.copy(pred_image)
    cv2.drawContours(temp_image, np.array(([final_contours[top_pool_id]])), -1, (0,0,255), cv2.FILLED)

    if bool_display_final_contour == True:
        cv2.imshow('Blood pool with the highest volume drawing', temp_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return top_pool_id, top_pool_deepest_point_id, top_pool_volume
        

            


