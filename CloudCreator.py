import cv2

before = cv2.imread("./Dataset/Processed/pc_mock/before.jpg")  
after = cv2.imread("./Dataset/Processed/pc_mock/after.jpg")  

before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB);
after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB); 



width = 330
height = 250
max_colour = 255*3

f=open("before.PCD", "a+")
for i in range(len(before)):
    for j in range(len(before[i])):
        cur_colour = int(before[i][j][0]) + int(before[i][j][1]) + int(before[i][j][2])
        cur_depth = max_colour - cur_colour
        f.write(str(i) + " " + str(j) + " " + str(cur_depth) + "\n")
f.close()

f=open("after.PCD", "a+")
for i in range(len(after)):
    for j in range(len(after[i])):
        cur_colour = int(after[i][j][0]) + int(after[i][j][1]) + int(after[i][j][2])
        cur_depth = max_colour - cur_colour
        f.write(str(i) + " " + str(j) + " " + str(cur_depth) + "\n")
f.close()
