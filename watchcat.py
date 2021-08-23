# -*- coding: utf-8 -*-
"""
OpenCV Motion Detector
@author: methylDragon
                                   .     .
                                .  |\-^-/|  .
                               /| } O.=.O { |\
                              /´ \ \_ ~ _/ / `\
                            /´ |  \-/ ~ \-/  | `\
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
import random
import numpy as np
import cv2
import pyautogui

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

# =============================================================================
# CORE PROGRAM
# =============================================================================


# Create capture object
cap = cv2.VideoCapture(5) # Flush the stream
cap.release()
cap = cv2.VideoCapture(0) # Then start the webcam

# Init frame variables
first_frame = None
next_frame = None

# Init display font and timeout counters
font = cv2.FONT_HERSHEY_SIMPLEX
delay_counter = 0
movement_persistent_counter = 0
n_frame = 0

screenshot = pyautogui.screenshot()
screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_width = 1280
out_height = 960
out = cv2.VideoWriter('screenshot'+str(random.randint(1,1000000))+".avi",fourcc, 20.0, (out_width,out_height))



# LOOP!
while True:
    n_frame = n_frame + 1
    # Set transient motion detected as false
    transient_movement_flag = False
    
    # Read frame
    ret, frame = cap.read()
    text = "Unoccupied"

    # If there's an error in capturing
    if not ret:
        print("CAPTURE ERROR")
        continue

   


    # take screenshot using pyautogui
    screenshot = pyautogui.screenshot()
    # since the pyautogui takes as a 
    # PIL(pillow) and in RGB we need to 
    # convert it to numpy array and BGR 
    # so we can write it to the disk
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Resize and save a greyscale version of the image
    frame = imutils.resize(frame, width = screenshot.shape[1])
   
    screenshot = cv2.resize(screenshot, (frame.shape[1], frame.shape[0]))   
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
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Fill in holes via dilate(), and find contours of the thesholds
    thresh = cv2.dilate(thresh, None, iterations = 2)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in contours:

        # Save the coordinates of all found contours
        (x, y, w, h) = cv2.boundingRect(c)
        
        # If the contour is too small, ignore it, otherwise, there's transient
        # movement
        if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
            transient_movement_flag = True
            
            # Draw a rectangle around big enough movements
#          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # The moment something moves momentarily, reset the persistent
    # movement timer.
    if transient_movement_flag == True:
        movement_persistent_flag = True
        movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

    
    #get current date and time
    x = datetime.datetime.now()

    #convert date and time to string
    dateTimeStr = str(x)
  
    # writing it to the disk using opencv
    #cv2.imwrite("image1.png", screenshot)


    # For if you want to show the individual video frames
    
    # Convert the frame_delta to color for splicing
    frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)
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

    cv2.putText(frame, str("Kvazikot (vsbaranov83@gmail.com) ") + dateTimeStr, (10,55), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
    stack_image = np.hstack((screenshot, frame))
    stack_image = cv2.resize(stack_image, (out_width, out_height))   

    if (n_frame % 5)==0:
        if (movement_persistent_counter < 100) and (movement_persistent_counter > 90): 
             #cv2.imwrite("scr"+str(random.randint(1,1000000))+".jpg", stack_image)
             out.write(stack_image)

    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        out.release()
        break
    else: 
        if ch & 0xFF == ord(' '):    
            cv2.imwrite("scr"+str(random.randint(1,1000000))+".jpg", stack_image)
            


    # Splice the two video frames together to make one long horizontal one


    cv2.imshow("frame", stack_image)
 


  

# Cleanup when closed
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
out.release()