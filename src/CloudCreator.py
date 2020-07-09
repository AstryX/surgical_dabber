import cv2
import numpy as np
import struct

before = cv2.imread("./Dataset/Processed/pc_mock/before.jpg")  
after = cv2.imread("./Dataset/Processed/pc_mock/after.jpg")
colour = cv2.imread("./Dataset/Processed/Img/img (161).jpg")  

before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB);
after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB); 
colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB); 

width = 330
height = 250
max_colour = 255*3

f=open("before.PCD", "a+")
for i in range(len(before)):
    for j in range(len(before[i])):
        cur_colour = int(before[i][j][0]) + int(before[i][j][1]) + int(before[i][j][2])
        cur_depth = max_colour - cur_colour
        f.write(str(i) + " " + str(j) + " " + str(cur_depth) + " " + str(struct.unpack('<f',bytearray([colour[i][j][0],colour[i][j][1],colour[i][j][2],0]))[0]) + "\n")
f.close()

f=open("after.PCD", "a+")
for i in range(len(after)):
    for j in range(len(after[i])):
        cur_colour = int(after[i][j][0]) + int(after[i][j][1]) + int(after[i][j][2])
        cur_depth = max_colour - cur_colour
        f.write(str(i) + " " + str(j) + " " + str(cur_depth) + " " + str(struct.unpack('<f',bytearray([colour[i][j][0],colour[i][j][1],colour[i][j][2],0]))[0]) + "\n")
f.close()
