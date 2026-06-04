
import cv2
import numpy as np
from tracker import EuclideanDistTracker
import math
import matplotlib.pyplot as plt
import csv
import pandas as pd
import time

#Data for the Velocity
fps = 5000
frame_width_px = 640
frame_height_py= 480 

real_width_A = 53.5977                                           # In cm
real_height_A = 40.198  

meters_per_pixel_x = (real_width_A/100)/ frame_width_px
meters_per_pixel_y = (real_height_A/100)/frame_height_py
centroids = []
frame_indices=[]
Tracker_ejecta= EuclideanDistTracker()                                     #Create tracker object( file in the same location as the py file)

csrt_tracker=None
boulder_initialized= False

video= cv2.VideoCapture(r"C:\Users\Admin\Desktop\Internship Documents\Open CV Documents\Merengues bowl1_C001S0001.avi")
# object_d=cv2.createBackgroundSubtractorMOG2(1000,40,False)             #Detection of the objects 
object_d= cv2.createBackgroundSubtractorKNN(1000,400, False)
#Start with the video play and looping
c=0
frame_count=0
while video.isOpened():
    r, frame= video.read()
    if r==True:
        filename =r"C:\Users\Admin\Desktop\Internship Documents\Open CV Documents\Boulder detection_front frame\Boulder Detection\Top View\New\frame"+str(c)+".png"
        h,w, _= frame.shape

#Boulder Number 5: Target for the Boulder travel determination.
        # print(h,w)                
        frame= cv2.resize(frame,(640,480))          #RoI for the fragments/ejecta
        # frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi= frame[20:200,0:400]
        # blur= cv2.GaussianBlur(roi, (5,5), 0)
        mask= object_d.apply(roi)
        _, mask= cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # remove specks
        mask = cv2.dilate(mask, kernel, iterations=2)  # fill gaps

        contours, _= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        detection=[]
        for cnt in contours:
            area= cv2.contourArea(cnt)
            # print(area)                                                              #To determine the area of the countour that helps in the area selection for the oebject detection
            if area>20:         
                x,y,w,h= cv2.boundingRect(cnt)
                cv2.drawContours(roi, [cnt], -1, (0,255,0), 1)
                # cv2.rectangle(roi, (x,y), (x+w,y+h), (0,255,0), 1)
                detection.append([x,y,w,h])
                #print(detection)
                # print(x,y,w,h) 
# Tracking of each debrie that is ejected out.(optional to use)
                Target_id= 10
                boxes_ids= Tracker_ejecta.update(detection)
                for box_id in boxes_ids:
                    x,y,w,h,id= box_id
                    if id== Target_id:
                        cv2.putText(roi, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
                        cv2.rectangle(roi, (x,y), (x+w,y+h), (0,255,0), 1)
#For the calculation of velocity, we direct the detection values to the pixel points and find the location:
                        if id== Target_id:
                            cx= x+w/2
                            cy= y+h/2
                            centroids.append((cx,cy))
                            frame_indices.append(frame_count)
                        else:
                            if centroids:
                                last_cx,last_cy= centroids[-1]
                                centroids.append((last_cx, last_cy))
                                frame_indices.append(frame_count)
        cv2.imshow("Mask", mask)
        cv2.imshow("ROI",roi)
        cv2.imshow("Window", frame)
        # cv2.imwrite(filename,frame)
        c=c+1
        frame_count+=1
        if cv2.waitKey(200) & 0xff==ord("z"):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()

velocities=[]
time_value1=[]
distances=[]
# Compute velocity between consecutive centroids
for i in range(1, len(centroids)):
    cx1, cy1 = centroids[i-1]
    cx2, cy2 = centroids[i]
    dx_px = cx2 - cx1
    dy_px = cy2 - cy1
    dx_m = dx_px * meters_per_pixel_x
    dy_m = dy_px * meters_per_pixel_y
    distance = math.hypot(dx_m, dy_m)
    dt = (frame_indices[i]-frame_indices[i-1])/ fps                                                             # consecutive frames
    if dt>0:
        velocity = distance / dt
        velocities.append(velocity)
        time_value1.append(frame_indices[i]/fps)                              #For equal interval of time, earlier it was (dt*i)
        distances.append(distance)
        print(f"Frame {frame_indices[i-1]} -> {frame_indices[i]}: "f"dx={dx_m:.4f} m, dy={dy_m:.4f} m, "f"distance={distance:.4f} m, velocity of Projectile ={velocity:.2f} m/s")  


time_value=[f"{t:.4f}" for t in time_value1]
distances_value=[f"{d:.4f}" for d in distances]
vvelocities_value=[f"{v:.2f}" for v in velocities]

timestamp=time.strftime("%Y%m%d_%H%M%S")
loaction=fr"C:\Users\Admin\Desktop\Internship Documents\Open CV Documents\Boulder detection_front frame\Boulder Detection\Top View\New\Data_{timestamp}.csv"

with open(loaction,mode='w', newline='') as f:
    writer=csv.writer(f,delimiter=';')
    writer.writerow(["Frame","Distance(m)", "Time(s)", "Velocity(m/s)"])
    for fr,d,t,v in zip(frame_indices[1:],distances_value,time_value,vvelocities_value):
        writer.writerow([fr,d,t,v])
            

#Plotting of the velocity profile for the frames captured.

df = pd.read_csv(loaction, sep=";")
df_filtered = df[(df["Frame"] >= 35) & (df["Frame"] <= 60)]

# Extract time and velocity
time_value1 = df_filtered['Time(s)']
velocity = df_filtered['Velocity(m/s)']

# Plot (scatter or line)
plt.figure(figsize=(10,6))
plt.plot(time_value1, velocity , marker='o', linestyle='-', color='blue', label=f"Velocity vs Time{Target_id}")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title(f"Velocity vs Time for boulder{Target_id}")
plt.grid(True)
plt.legend()
plt.show()

