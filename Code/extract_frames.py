import cv2
 
video = cv2.VideoCapture('C:/Users/Mohammad Khorasani/Desktop/Videos/24kph.mp4')

i = 0

while(video.isOpened()):
    
    ret, frame = video.read()
    
    if ret == False:
        break
    
    cv2.imwrite('C:/Users/Mohammad Khorasani/Desktop/24kph/'+str(i)+'.png',frame)
    i = i + 1
 
video.release()
cv2.destroyAllWindows()
