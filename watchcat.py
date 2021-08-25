# -*- coding: utf-8 -*-
"""
OpenCV Motion Detector
@author: methylDragon
                                   .     .
                                .  |\-^-/|  .
                               /| } O.=.O { |\
                              /Вґ \ \_ ~ _/ / `\
                            /Вґ |  \-/ ~ \-/  | `\
                            |   |  /\\ //\  |   |
                             \|\|\/-""-""-\/|/|/
                                     ______/ /
                                     '------'
                       _   _        _  ___
             _ __  ___| |_| |_ _  _| ||   \ _ _ __ _ __ _ ___ _ _
            | '  \/ -_)  _| ' \ || | || |) | '_/ _` / _` / _ \ ' \
            |_|_|_\___|\__|_||_\_, |_||___/|_| \__,_\__, \___/_||_|
                               |__/                 |___/
            -------------------------------------------------------
                           github.com/methylDragon
References/Adapted From:
https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
Description:
This script runs a motion detector! It detects transient motion in a room
and said movement is large enough, and recent enough, reports that there is
motion!
Run the script with a working webcam! You'll see how it works!

modified by Kvazikot

"""

import imutils
import cv2
import numpy as np
import datetime
import time
import random
import numpy as np
import cv2
import pyautogui
import os
from threading import Thread

# =============================================================================
# USER-SET PARAMETERS
# =============================================================================

# Number of frames to pass before changing the frame to compare the current
# frame against
FRAMES_TO_PERSIST = 10
# Minimum boxed area for a detected motion to count as actual motion
# Use to filter out noise or small objects
MIN_SIZE_FOR_MOVEMENT = 2000
# Minimum length of time where no motion is detected it should take
#(in program cycles) for the program to declare that there is no movement
MOVEMENT_DETECTED_PERSISTENCE = 100
SCREENSHOT_ENABLED = 1
CAMERA0_ENABLED = 1
out_width = 1280
out_height = 960

def warpPerspVFX(img):
    x = 0
    y = 0
    w = frame.shape[1]
    h = frame.shape[0]
    corners = [[x,y], [w, y], [w, h], [x, h]]    
    x = x + random.randint(1,10)
    w = w - random.randint(1,10)
    new = np.float32([[x,y], [w, y], [w, h], [x, h]]   )
    M = cv2.getPerspectiveTransform(np.float32(corners), new)
    img = cv2.warpPerspective(img, M, ( frame.shape[1], frame.shape[0]))
    return img




class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame0) = self.stream.read()
        self.frame = np.zeros((out_height,out_width,3), dtype=np.uint8)
        self.stopped = False
    def start(self):
        Thread(target=self.get, args=()).start()
        return self
    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                t0 = time.time_ns() 
                (self.grabbed, self.frame0) = self.stream.read()
                self.frame = self.processing(self.frame0)
                lastFrameTime = (time.time_ns() - t0) / (10 ** 9)
                print('lastFrameTime Distortion ' + str(lastFrameTime))

                time.sleep(0.4) 
                #print("get(self)")
                
    def processing(self, frame):
            frame = cv2.resize(frame, (out_width, out_height))
            frame = cv2.GaussianBlur(frame, (41, 41), 0)

            # warp coordinates from x,y -> f(phi,r)
            center = (frame.shape[1]/2, frame.shape[0]/2)

            red_channel, green_channel, blue_channel = cv2.split(frame)


            red_channel = cv2.warpPolar(red_channel, (frame.shape[1], frame.shape[0]),
                                  center, frame.shape[1],
                                  cv2.WARP_POLAR_LINEAR )  

            #print(frame)

            # add noise to polar coordinates
            noise = np.random.normal(12, (12), (red_channel.shape[0], red_channel.shape[1]))        
            noise = noise.astype(np.uint8)
            #print(noise)

            blur_level = random.randint(40,60)
            if (blur_level % 2 ) == 0:
                blur_level = blur_level + 1

            red_channel = red_channel + noise
            red_channel = np.where((noise > 12), red_channel-12, red_channel).astype('uint8')    
            red_channel = cv2.GaussianBlur(red_channel, (blur_level, blur_level), 0)

            red_channel = warpPerspVFX(red_channel)
            #cv2.imshow("noise", noise)
            center = (center[0] - 1, center[1] + 1)
            red_channel = cv2.warpPolar(red_channel, (frame.shape[1], frame.shape[0]),
                                  center, frame.shape[1],
                                   cv2.WARP_INVERSE_MAP )  


            #print(frame)
            frame = cv2.merge((red_channel, red_channel, red_channel))
            return frame

    def stop(self):
        self.stopped = True

# =============================================================================
# CORE PROGRAM
# =============================================================================


# Create capture object

video_getter = VideoGet(0).start()
#cap = cv2.VideoCapture('test_video.mp4')
#start_frame_number = 50000
#cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
#fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
fps = 25
frame = np.zeros((out_height,out_width,3), dtype=np.uint8)
# Init frame variables
first_frame = None
next_frame = None
stack_image = None
movement_persistent_flag = False
read_frame_flag = True

# Init display font and timeout counters
font = cv2.FONT_HERSHEY_SIMPLEX
delay_counter = 0
movement_persistent_counter = 0
n_frame = 0

#latest screenshot number
latest_filenum = 0
files = os.listdir()
for file in files:
    if file.find("screenshot_") is not -1:
        parts = file.split('.')
        parts = parts[0]
        parts = parts.split('_')
        latest_filenum = int(parts[1])

screenshot = pyautogui.screenshot()
screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('screenshot_'+str(latest_filenum+1)+".avi",fourcc, 20.0, (out_width,out_height))
#frame = np.zeros((out_height,out_width,3), dtype=np.uint8)

#get current date and time
x = datetime.datetime.now()
y = x.replace(year=x.year + random.randint(-5,5))
y = y.replace(month=x.month + random.randint(-2,2))

#convert date and time to string
dateTimeStr = str(y)

def my_stack(stack):
    if CAMERA0_ENABLED:
        largeImage = stack[0]        
        size = (largeImage.shape[1], largeImage.shape[0])
        new_size = (round(stack[0].shape[1]/8), round(stack[0].shape[0]/8),3)
        #print('new_size ' + str(new_size))
        smallImage = np.zeros(new_size, dtype=np.uint8)

        #smallImage = cv2.resize(stack[0], (new_size[0],new_size[1]))
        ofs = (stack[0].shape[1]-new_size[1], stack[0].shape[0]-new_size[0])
        #print('offsets ' + str(ofs))
        #largeImage[ofs[0]:size[0],ofs[1]:size[1]] = smallImage.copy()
        #cv2.imshow('largeImage',largeImage)
    return np.hstack((stack[0],stack[1]))


# LOOP!
while True:
    t0 = time.time_ns()
    n_frame = n_frame + 1
    # Set transient motion detected as false
    transient_movement_flag = False
    

    # take screenshot using pyautogui
    screenshot = pyautogui.screenshot()
    # since the pyautogui takes as a 
    # PIL(pillow) and in RGB we need to 
    # convert it to numpy array and BGR 
    # so we can write it to the disk
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Resize and save a greyscale version of the image
  
    screenshot = cv2.resize(screenshot, (out_width, out_height))   
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Blur it to remove camera noise (reducing false positives)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # If the first frame is nothing, initialise it
    if first_frame is None: first_frame = gray    

    delay_counter += 1

    # Otherwise, set the first frame to compare as the previous frame
    # But only if the counter reaches the appriopriate value
    # The delay is to allow relatively slow motions to be counted as large
    # motions if they're spread out far enough
    if delay_counter > FRAMES_TO_PERSIST:
        delay_counter = 0
        first_frame = next_frame

        
    # Set the next frame to compare (the current frame)
    next_frame = gray

    # Compare the two frames, find the difference
    frame_delta = cv2.absdiff(first_frame, next_frame)
    # Convert the frame_delta to color for splicing
    frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    tot_pixel = thresh.size 
    non_zero_pixels = np.count_nonzero(thresh)
    percentage = round(non_zero_pixels * 100 / tot_pixel, 2)
    #print('non_zero_pixels '+str(percentage))
    # Fill in holes via dilate(), and find contours of the thesholds
    if percentage > 0.01:
        transient_movement_flag = True

    #thresh = cv2.dilate(thresh, None, iterations = 2)
    #contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ## loop over the contours
    #for c in contours:

    #    # Save the coordinates of all found contours
    #    (x, y, w, h) = cv2.boundingRect(c)
        
    #    # If the contour is too small, ignore it, otherwise, there's transient
    #    # movement
    #    if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
    #        transient_movement_flag = True
           

    # The moment something moves momentarily, reset the persistent
    # movement timer.
    if transient_movement_flag == True:
        movement_persistent_flag = True
        movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

    

    
    if (movement_persistent_counter > 90 or movement_persistent_counter == 2): 
        read_frame_flag = True
    else:
        read_frame_flag = False
    

    if read_frame_flag:
       if video_getter.grabbed:
          frame = video_getter.frame

    #ret, frame = cap.read()
   

 
    # writing it to the disk using opencv
    #cv2.imwrite("image1.png", screenshot)


    # For if you want to show the individual video frames
    
        # Interrupt trigger by pressing q to quit the open CV program
  

    cv2.putText(frame, str("press space for screenshot"), (10,75), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
    

    # As long as there was a recent transient movement, say a movement
    # was detected    
    if movement_persistent_counter > 0:
        text = "Movement Detected " + str(movement_persistent_counter)
        movement_persistent_counter -= 1
        #cv2.imwrite("scr"+str(random.randint(1,1000000))+".png", stack_image)
    else:
        text = "No Movement Detected"

           # Print the text on the screen, and display the raw and processed video 
    # feeds
    cv2.putText(frame, str(text), (10,35), font, 0.75, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(frame, str("QuasiCode1 ") + dateTimeStr, (10,55), font, 0.75, (255,255,255), 2, cv2.LINE_AA)

    ch = cv2.waitKey(1)

    if ch & 0xFF == ord('1'):
        SCREENSHOT_ENABLED = not SCREENSHOT_ENABLED
        movement_persistent_counter = 2
    if ch & 0xFF == ord('2'):
        CAMERA0_ENABLED = not CAMERA0_ENABLED
        movement_persistent_counter = 2

    if SCREENSHOT_ENABLED == 0:
        cv2.rectangle(screenshot,(0,0),(screenshot.shape[1],screenshot.shape[0]),(0,0,0),cv2.FILLED)
        print("disable screenshot")

    if CAMERA0_ENABLED == 0:
        cv2.rectangle(frame,(0,0),(frame.shape[1],frame.shape[0]),(0,0,0),cv2.FILLED)
        print("disable camera")

     # Malevich instead of date        
    cv2.rectangle(screenshot,(screenshot.shape[1]-60,screenshot.shape[0]-200),(screenshot.shape[1]-10,screenshot.shape[0]-10),(0,0,0),cv2.FILLED)




    #stack_image = np.hstack((screenshot, frame))
    stack_image = my_stack([screenshot, frame])
    stack_image = cv2.resize(stack_image, (out_width, out_height))       
    out.write(stack_image)
             
    lastFrameTime = (time.time_ns() - t0) / (10 ** 9)
    print('lastFrameTime ' + str(lastFrameTime))
    time.sleep(0.1)

    if ch & 0xFF == ord('q'):
        out.release()
        video_getter.stop()
        break
    else: 
        if ch & 0xFF == ord(' '):    
            cv2.imwrite("scr"+str(random.randint(1,1000000))+".jpg", stack_image)
            
    if stack_image is not None:
        if read_frame_flag is True:
            cv2.rectangle(stack_image,(0,0),(400,200),(0,0,255),cv2.FILLED)
        else:
            cv2.rectangle(stack_image,(0,0),(400,200),(0,0,0),cv2.FILLED)
        cv2.putText(stack_image, str(movement_persistent_counter), (10,120), font, 4.75, (255,255,255), 6, cv2.LINE_AA)            
        #cv2.putText(stack_image, str(text), (10,100), font, 0.75, (255,255,255), 2, cv2.LINE_AA)            
        cv2.imshow("frame", stack_image)
    else:
        cv2.imshow("frame", frame)            

    # Splice the two video frames together to make one long horizontal one


    
 


  

# Cleanup when closed
cv2.waitKey(0)
cv2.destroyAllWindows()
video_getter.stop()
out.release()