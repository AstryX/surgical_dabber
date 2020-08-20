#!/usr/bin/env python
import json
import math
import time
from Predictor import predictImageLabels, findOptimalDestination
from Helper import loadPointCloud

params_path = "params.json"
predict_num = 200
image_size_rows = 250
image_size_cols = 330
bool_display_final_contour = True
im_path = "Dataset/Processed/"

with open(params_path) as json_data_file:  
    data = json.load(json_data_file)
    if 'predict_num' in data:
        predict_num = int(data['predict_num'])
    if 'image_size_rows' in data:
        image_size_rows = data['image_size_rows']
    if 'image_size_cols' in data:
        image_size_cols = data['image_size_cols']
    if 'im_path' in data:
        im_path = data['im_path']
    if 'bool_display_final_contour' in data:
        bool_display_final_contour = data['bool_display_final_contour']

before_path = im_path+'pc_mock/before_3.PCD'
after_path = im_path+'pc_mock/after_3.PCD'

before = loadPointCloud(before_path)
after = loadPointCloud(after_path)
print('Before and after shape after cloud loading:')
print(before.shape)
print(after.shape)

before_z = before['z']
after_z = after['z']

time_benchmark = time.time()
pred_labels, final_contours, loaded_image = predictImageLabels(params_path, predict_num, im_path, '')
print("Run time of Image pixel prediction: " + str(time.time() - time_benchmark) + " seconds.")
time_benchmark = time.time()
top_pool_id, top_pool_deepest_point_id, top_pool_volume = findOptimalDestination(before_z, after_z,
    image_size_rows, image_size_cols, pred_labels, final_contours, loaded_image, bool_display_final_contour)
print("Run time of Optimal Dab Destination computation: " + str(time.time() - time_benchmark) + " seconds.")