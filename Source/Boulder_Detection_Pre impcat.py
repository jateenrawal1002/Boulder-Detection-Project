import cv2
import numpy as np
import math
import pandas as pd
image= cv2.imread(r"c:\Users\Admin\Desktop\Internship Documents\Open CV Documents\Boulder detection_front frame\Post and Pre impact\20231025_175315.jpg")
# image = cv2.imread(r"C:\Users\Admin\Desktop\Internship Documents\Open CV Documents\Boulder detection_front frame\Post and Pre impact\20231025_123613.jpg")
image = cv2.resize(image, (1050, 580))
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
new_img = cv2.medianBlur(gray_img, 5)

#pre impact image data.
# min_distance=32
# param01=82
# param02=31
# min_radius=17
# max_radius=37

#Post Impact Data 
# min_distance=32
# param01=82
# param02=31
# min_radius=17
# max_radius=37


def one(x):
    pass

cv2.namedWindow("Boulder_Detection")
cv2.createTrackbar("min_distance","Boulder_Detection" , 30,255, one) 
cv2.createTrackbar("param01","Boulder_Detection" , 100,200, one)
cv2.createTrackbar("param02", "Boulder_Detection", 30,100, one)
cv2.createTrackbar("min_radius","Boulder_Detection" , 15,50, one)
cv2.createTrackbar("max_radius", "Boulder_Detection",40,100,one)

while True:

    min_distance=cv2.getTrackbarPos("min_distance", "Boulder_Detection")
    param01=cv2.getTrackbarPos("param01", "Boulder_Detection")
    param02=cv2.getTrackbarPos("param02", "Boulder_Detection")
    min_radius=cv2.getTrackbarPos("min_radius", "Boulder_Detection")
    max_radius=cv2.getTrackbarPos("max_radius", "Boulder_Detection")

    if min_distance <1: min_distance=1
    if param02 <1:param02=1

    circles = cv2.HoughCircles(new_img,cv2.HOUGH_GRADIENT,dp=1,minDist=min_distance,param1=param01,param2=param02,minRadius=min_radius,maxRadius=max_radius)
    
    output_image = image.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle_id=1
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(output_image, center, radius, (0, 255, 0), 2)
            area= math.pi*(radius**2)
            id_text= f"ID:{circle_id}"


            cv2.putText(output_image, id_text,(center[0]-20,center[1]-30), cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),1)    

            circle_id+=1

        cv2.imshow('Boulder_Detection', output_image)
        if cv2.waitKey(1) & 0xff==ord("z"):
            break

# cv2.waitKey(0)
cv2.destroyAllWindows()

