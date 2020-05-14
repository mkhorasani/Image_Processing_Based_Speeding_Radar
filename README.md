## Image Processing Based Vehicle Number Plate Detection and Speeding Radar

The scope of this project has been to implement an image processing-based traffic radar that detects vehicle number plates and subsequently measures the instantaneous vehicle speed. This application of computational photography/image processing was selected in order to develop an open source and cost-effective alternative to current speeding radar systems that can carry a price tag upwards of $6,500 per unit. As an open source technique, this will enable local authorities, municipalities, and facilities to implement their own low-cost ($1,700) and convenient traffic monitoring systems with off the shelf devices and equipment.

Instructions:

1. Procure a camera with the following specifications: 24 fps, 1080p, 1/60 seconds exposure time
2. Place camera immediately to the side of the road with an inclination of 0°, a horizontal angle 
   of approximately 20° towards the road and an elevation of approximately 2 feet off the ground
3. Record video of moving vehicle 
4. Upload video in any format i.e. mp4 to root directory
5. Run extract_frames.py to extract video froms to root directory
6. Run resize_numbers.py to create templates of all numbers and sizes and place them in root
   directory
7. Run main.py by adjusting line 465 to root directory location and wait for results to be saved
   in root directory
8. Optionally you may run frames_to_video.py to rcreate frames from main.py into a video.


Links to number plate and speed detection videos:

-	Video at 24fps: https://www.youtube.com/watch?v=otAKOXekY5w
-	Video at 12fps: https://www.youtube.com/watch?v=eGuRIqdugiU

--------------------------------------------------------------------------------------------------
