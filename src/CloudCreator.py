import cv2
import numpy as np
import struct

before = cv2.imread("./Dataset/Processed/pc_mock/before_3.jpg")  
after = cv2.imread("./Dataset/Processed/pc_mock/after_3.jpg")
colour = cv2.imread("./Dataset/Processed/Img/img (200).jpg")  

before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB);
after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB); 
colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB); 

width = 330.0
height = 250.0
max_colour = 255*3.0

f=open("before_3.PCD", "a+")
for i in range(len(before)):
    for j in range(len(before[i])):
        cur_colour = int(before[i][j][0]) + int(before[i][j][1]) + int(before[i][j][2])
        cur_depth = 1 - cur_colour / max_colour
        new_i = (float(i) / height) * 2 - 1.0
        new_j = (float(j) / width) * 2 - 1.0
        f.write(str(new_i) + " " + str(new_j) + " " + str(cur_depth) + " " + str(struct.unpack('<f',bytearray([colour[i][j][2],colour[i][j][1],colour[i][j][0],0]))[0]) + "\n")
f.close()

f=open("after_3.PCD", "a+")
for i in range(len(after)):
    for j in range(len(after[i])):
        cur_colour = int(after[i][j][0]) + int(after[i][j][1]) + int(after[i][j][2])
        cur_depth = 1 - cur_colour / max_colour
        new_i = (float(i) / height) * 2 - 1.0
        new_j = (float(j) / width) * 2 - 1.0
        f.write(str(new_i) + " " + str(new_j) + " " + str(cur_depth) + " " + str(struct.unpack('<f',bytearray([colour[i][j][2],colour[i][j][1],colour[i][j][0],0]))[0]) + "\n")
f.close()
