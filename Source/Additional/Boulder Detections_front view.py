#======================================================================================================
#===============THE EJECTA DETECTION FROM THE FRONT VEIEW==============================================
# =====================Boulder Detection=====================================================
import cv2
import numpy as np
from tracker import EuclideanDistTracker
import math
import matplotlib.pyplot as plt
import csv
import pandas as pd
import time

# #Data for the Velocity
fps = 5000
frame_width_px = 640
frame_height_py= 480 

real_width_A = 53.5977                                           # In cm
real_height_A = 40.198  

meters_per_pixel_x = (real_width_A/100)/ frame_width_px
meters_per_pixel_y = (real_height_A/100)/frame_height_py

#We have enlisted some tracking ID's that we need to track. We store the values of the centeroids and frame indices in the disct to get them later.
Tracking_id = range(0,300)
centroids_values = {tid: [] for tid in Tracking_id}
frame_indices_values = {tid: [] for tid in Tracking_id}

Tracker_ejecta= EuclideanDistTracker()                                         #Create tracker object( file in the same location as the py file)

video= cv2.VideoCapture(r"C:\Users\Admin\Desktop\Internship Documents\Open CV Documents\Merengues-bowl_195bar_delrin_qs_vert.avi")
# object_d=cv2.createBackgroundSubtractorMOG2(1000,40,False)             #Detection of the objects 
object_d= cv2.createBackgroundSubtractorKNN(1000,400, False)
c=0
frame_count=0
warmup_frames=50
while video.isOpened():
    r, frame= video.read()
    if r==True:
        filename =r"C:\Users\Admin\Desktop\Internship Documents\Open CV Documents\Boulder detection\Boulder Detection\Boulder 29\Front View\Frames\frame"+str(c)+".png"
        h,w, _= frame.shape   
        frame_gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray=cv2.GaussianBlur(frame,(3,3),0)
        learning_rate=-1 if frame_count<warmup_frames else 0
        mask= object_d.apply(frame,learningRate=learning_rate)


        _, mask= cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # remove specks
        mask = cv2.dilate(mask, kernel, iterations=2)  # fill gap

        contours, _= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detection=[]
        for cnt in contours:
            area= cv2.contourArea(cnt)
            # print(area)                                                             #To determine the area of the countour that helps in the area selection for the oebject detection
            if area<20 or area>20000:
                continue
                     
            x,y,w,h= cv2.boundingRect(cnt)


            hull=cv2.convexHull(cnt)
            hull_area= cv2.contourArea(hull) if len(hull)>=3 else 1.0
            solidity= float(area)/hull_area if hull_area>0 else 0
            rect_area= float(w*h)
            extent= float(area)/rect_area if rect_area>0 else 0
            print(f"area={area:.1f}, solidity={solidity:.2f}, E={extent:.2f}")

            if solidity<0.5:
                continue

            if extent<0.15:
                continue
            detection.append([x,y,w,h])
            cv2.drawContours(frame, [cnt], -1, (0,255,0), 1)
            #cv2.rectangle(roi, (x,y), (x+w,y+h), (0,255,0), 1)
            #print(detection)
            #print(x,y,w,h)

# Detecting the object and checks the new position with the previous position of the frame.
        boxes_ids = Tracker_ejecta.update(detection)
        for x, y, w, h, id in boxes_ids:
            if id in Tracking_id:
                cv2.putText(frame, str(id), (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 1)
                cx = x + w/2.0
                cy = y + h/2.0
                centroids_values[id].append((cx, cy))
                print(f"ID:{id},Centroid:({cx},{cy})")
                frame_indices_values[id].append(frame_count)
        cv2.imshow("Mask", mask)
        cv2.imshow("Window", frame)
        # cv2.imwrite(filename,frame)
        c=c+1
        frame_count+=1
        if cv2.waitKey(1000) & 0xff==ord("z"):
            break
      
video.release()
cv2.destroyAllWindows()


# # Velocity, distance, time and frame determination of the individual ID's

distances_dict = {tid: [] for tid in Tracking_id}           # Making th dict to store the value of the data for each ID.
velocities_dict = {tid: [] for tid in Tracking_id}
time_dict = {tid: [] for tid in Tracking_id}
rows=[]

for tid in Tracking_id:
    centroids = centroids_values[tid]
    frames = frame_indices_values[tid]

    if len(centroids) < 2:                  # To check that we have atleast two data for the centroids for the velocity determination
        continue

    for i in range(1, len(centroids)):
        cx1, cy1 = centroids[i-1]
        cx2, cy2 = centroids[i]

        # Pixel differences
        dx_px = cx2 - cx1
        dy_px = cy2 - cy1

        # Convert to meters
        dx_m = dx_px * meters_per_pixel_x
        dy_m = dy_px * meters_per_pixel_y

        distance = math.hypot(dx_m, dy_m)
        dt = (frames[i] - frames[i-1]) / fps

        if dt > 0:
            velocity = distance / dt
            t_sec = frames[i] / fps
            distances_dict[tid].append(distance)
            velocities_dict[tid].append(velocity)
            time_dict[tid].append(t_sec)

            print(f"ID={tid} | Centroid= ({cx2},{cy2})| Frame {frames[i-1]}->{frames[i]} | "f"dx={dx_m:.4f} m, dy={dy_m:.4f} m, "f"distance={distance:.4f} m, velocity={velocity:.2f} m/s, time={t_sec:.4f} s")
            rows.append([tid, f"{frames[i-1]}->{frames[i]}",f"({cx2:.2f},{cy2:.2f})" ,f"{distance:.4f}", f"{t_sec:.4f}", f"{velocity:.2f}"])
            
timestamp=time.strftime("%Y%m%d_%H%M%S")
location=fr"C:\Users\Admin\Desktop\Internship Documents\Open CV Documents\Boulder detection\Boulder Detection\Boulder 29\Front View\data_{timestamp}.csv"

with open(location,mode='w', newline='') as f:
    writer=csv.writer(f,delimiter=';')
    writer.writerow(["ID","FramePair","Centroid(x,y)","Distance(m)", "Time(s)", "Velocity(m/s)"])
    writer.writerows(rows)

stored_val = pd.read_csv(location, sep=";")
stored_val["ID"] = stored_val["ID"].astype(int)
stored_val["Velocity(m/s)"] = stored_val["Velocity(m/s)"].astype(float)
stored_val["Time(s)"] = stored_val["Time(s)"].astype(float)

frame_min, frame_max = 0,10
t_min,t_max= frame_min/fps, frame_max/fps
ids_to_plot = [0,2,4]

plt.figure(figsize=(15, 12))

for tid in ids_to_plot:
    df_filtered = stored_val[(stored_val["ID"] == tid) & (stored_val["Time(s)"] >= t_min) & (stored_val["Time(s)"] <= t_max)] 

    if not df_filtered.empty:
        plt.plot(df_filtered["Time(s)"], df_filtered["Velocity(m/s)"], marker='o', linestyle='-', color='royalblue',
         markersize=6, linewidth=2, markerfacecolor='orange', markeredgecolor='black', label=f"ID {tid}")
plt.xlabel("Time (s)",fontsize=12, fontweight='bold')
plt.ylabel("Velocity (m/s)",fontsize=12, fontweight='bold')
plt.title(f"Velocity vs Time for IDs {ids_to_plot} (Frames {frame_min}-{frame_max})",fontsize=12, fontweight='bold')
plt.grid(True,linestyle='--', alpha=0.6)
plt.legend()
plt.show()



#======================================================================================================
#===============THE PROJECTILE DETECTION FROM THE FRONT VEIEW==========================================

import cv2
import numpy as np
from tracker import EuclideanDistTracker
import math
import matplotlib.pyplot as plt
import csv

#Data for the Velocity
fps = 5000
frame_width_px = 640  
frame_height_py= 480

real_width_A = 52.484                                 # In cm
real_height_A = 39.363   
real_width_B =  53.5977
real_height_B = 40.198

meters_per_pixel_x = (real_width_B/100)/ frame_width_px
meters_per_pixel_y = (real_height_B/100)/frame_height_py
centroids = []
frame_inidices=[]
tracker= EuclideanDistTracker()                                         #Create tracker object( file in the same location as the py file)

video= cv2.VideoCapture(r"c:\Users\Admin\Desktop\Internship Documents\Open CV Documents\Merengues-bowl_195bar_delrin_qs_vert.avi")              #Detection of the objects 
object_d= cv2.createBackgroundSubtractorKNN(1000,400, False)
#Start with the video play and looping
c=0
frame_count=0
while video.isOpened():
    r, frame= video.read()
    if r==True:
        filename =r"C:\Users\Admin\Desktop\Internship Documents\Open CV Documents\Video of the Experiment\Images of the side view\Images (640x480) Projectile Detecton\Frame"+str(c)+".png"
        h,w, _= frame.shape
        # print(h,w)
        roi= frame[25:250,150:400]                #ROI for the projectile 
        # roi_1= frame[100:300,160:300]                #RoI for the fragments/ejecta
        # mask= object_d.apply(frame)
        mask= object_d.apply(roi)
        # mask= object_d.apply(roi_1)
        # mask=object_d.apply(frame)
        _, mask= cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)                                           
        contours, _= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        detection=[]
        for cnt in contours:
            area= cv2.contourArea(cnt)
            # print(area)                                                              #To determine the area of the countour that helps in the area selection for the oebject detection
            if area>350:              
                cv2.drawContours(roi, [cnt], -1, (0,255,0), 1)
                # cv2.drawContours(roi_1, [cnt], -1, (0,255,0), 1)
                # cv2.drawContours(frame,[cnt],-1,(0,255,0),1)
                x,y,w,h= cv2.boundingRect(cnt)
                # cv2.rectangle(roi, (x,y), (x+w,y+h), (0,255,0), 1)
                # cv2.rectangle(roi_1, (x,y), (x+w,y+h), (0,255,0), 1)
                # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)
                #cv2.circle(roi, (((x+w/2)), ((y+h)/2)), h/2, (255,0,0),1)             #To work on this.(optional)
                detection.append([x,y,w,h])
                #print(detection)
                # print(x,y,w,h) 
#Tracking of each debrie that is ejected out.(optional to use)
                boxes_ids= tracker.update(detection)
                for box_id in boxes_ids:
                    x,y,w,h,id= box_id
                    # if id==target_id:
                    cv2.rectangle(roi,(x,y), (x+w,y+h), (0,0,255), 1)
                    #cv2.putText(roi_1, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
                    # print(x,y,w,h,id)
#For the calculation of velocity, we direct the detection values to the pixel points and find the location:
            if detection:
                    x,y,w,h = detection[0]
                    cx= x+w/2
                    cy= y+h/2
                    centroids.append((cx,cy))
                    frame_inidices.append(frame_count) 
        cv2.imshow("Mask", mask)
        cv2.imshow("ROI",roi)
        # cv2.imshow("ROI_1", roi_1)
        cv2.imshow("Window", frame)
        # cv2.imwrite(filename,frame)
        c=c+1
        frame_count+=1
        if cv2.waitKey(1000) & 0xff==ord("z"):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()

velocities=[]
time=[]
projectile_sizes=[]
# Compute velocity between consecutive centroids
for i in range(1, len(centroids)):
    cx1, cy1 = centroids[i-1]
    cx2, cy2 = centroids[i]
    dx_px = cx2 - cx1
    dy_px = cy2 - cy1
    dx_m = dx_px * meters_per_pixel_x
    dy_m = dy_px * meters_per_pixel_y
    distance = math.hypot(dx_m, dy_m)
    dt = (frame_inidices[i]-frame_inidices[i-1])/ fps                                                             # consecutive frames
    if dt>0:
        velocity = distance / dt
        velocities.append(velocity)

        time.append(frame_inidices[i]/fps)                              #For equal interval of time, earlier it was (dt*i)
        print(f"Frame {frame_inidices[i-1]} -> {frame_inidices[i]}: "f"dx={dx_m:.4f} m, dy={dy_m:.4f} m, "f"distance={distance:.4f} m, velocity of Projectile ={velocity:.2f} m/s")

# # Converting the velocity data into a csv file for the further use:

# with open ("Velocity profile data.txt","w", newline="") as f:
#     writer= csv.writer(f)
#     # ourlist= [[dx_m],[dy_m],[distance],[velocity]]
#     ourlist= [[velocity]]
#     writer.writerows(ourlist)


# Velocity Vs the Time for the travel of the projectile.
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))  # Wider figure
plt.plot(time, velocities, marker='o', linestyle='-', color='royalblue',
         markersize=6, linewidth=2, markerfacecolor='orange', markeredgecolor='black')

# Labels and title
plt.xlabel("Time (s)", fontsize=12, fontweight='bold')
plt.ylabel("Velocity (m/s)", fontsize=12, fontweight='bold')
plt.title("Velocity vs Time", fontsize=14, fontweight='bold')

# Grid and background
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().set_facecolor("#f9f9f9")  # Light background behind plot

# Axis style
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()

plt.show()