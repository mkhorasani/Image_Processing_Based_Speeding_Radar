import cv2
import numpy as np
import matplotlib.pyplot as plt

six = cv2.imread('C:/Users/Mohammad Khorasani/Desktop/night seven/night seven.png',0)

for i in range(6,30):

    sv = cv2.resize(six,(int(i*len(six[0])*0.1),int(i*len(six)*0.1)))
    svp = cv2.cvtColor(sv,cv2.COLOR_BGR2RGB)
    plt.imsave('C:/Users/Mohammad Khorasani/Desktop/night seven/%s.png' % (i),svp)
