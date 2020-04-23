import cv2
import numpy as np
import os
from os.path import isfile, join
import re

frames_folder = 'C:/Users/Mohammad Khorasani/Desktop/24kph/'
output_folder = 'C:/Users/Mohammad Khorasani/Desktop/video.avi'

frames_array = []
files = [f for f in os.listdir(frames_folder) if isfile(join(frames_folder, f))]

#Sorting the filenames in ascending frame order

files.sort(key=lambda f: int(re.sub('\D', '', f)))


for i in range(0,len(files)):
    filename = frames_folder + files[i]

    #Reading each frame
    
    img = cv2.imread(filename)
    img = cv2.resize(img, (1920,1080))

    height, width, layers = img.shape
    size = (width,height)
    
    #Inserting frames into an array
    
    frames_array.append(img)
    
output = cv2.VideoWriter(output_folder,cv2.VideoWriter_fourcc(*'DIVX'), 12, size)

for i in range(0,len(frames_array)):
    
    #Converting array into video
    
    output.write(frames_array[i])
    
output.release()

