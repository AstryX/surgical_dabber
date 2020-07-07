import cv2
import json
from Helper import loadPointCloud

image_size_rows = 250
image_size_cols = 330

param_name = "params.json"
with open(param_name) as json_data_file:
		
    data = json.load(json_data_file)
    
    if 'image_size_rows' in data:
        image_size_rows = data['image_size_rows']
    if 'image_size_cols' in data:
        image_size_cols = data['image_size_cols']

before = loadPointCloud('./Dataset/Processed/pc_mock/before.PCD', image_size_rows, image_size_cols)
after = loadPointCloud('./Dataset/Processed/pc_mock/after.PCD', image_size_rows, image_size_cols)

print('Before and after shape after cloud loading:')
print(before.shape)
print(after.shape)

before_z = before[:,2]
after_z = after[:,2]

temp_image = np.copy(draw_image)
gray_mask = cv2.cvtColor(passed_mask, cv2.COLOR_BGR2GRAY)
print(print_text)
contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
