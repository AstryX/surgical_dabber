import cv2

imArray = []
numImg = 200
for i in range(numImg):
    fileName = "./Dataset/rawimg (" + str(i+1) + ").jpg"
    img = cv2.imread(fileName)  
    imArray.append(img)

print('Numimgs:')
print(len(imArray))    

width = int(330)
height = int(250)
dim = (width, height)
for i in range(numImg):  
    print('Processing img nr. ' + str(i+1))
    fileName = "./Dataset/Processed/Img/img (" + str(i+1) + ").jpg"
    resized = cv2.resize(imArray[i], dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(fileName, resized)

