import cv2
import cv2.cv as cv
import numpy as np
img = cv2.imread('/home/suraj/Repositories/LemonRecognition/lemon.jpeg')
print np.shape(img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv = cv2.blur(hsv,(5,5))
cv2.imshow('original', hsv)
cv2.waitKey(0)
im = hsv[:,:,1]
ret, bw = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('bw',bw)
cv2.waitKey(0)
contour, hierarchy = cv2.findContours(bw, cv.CV_RETR_TREE, cv.CV_CHAIN_APPROX_SIMPLE)
cnt = contour[0]
(x,y),radius = cv2.minEnclosingCircle(cnt)
cv2.circle(bw, (int(x), int(y)), int(radius), (255, 0, 0), 2)
cv2.imshow('bw',bw)
cv2.waitKey(0)
circles = cv2.HoughCircles(bw, method=cv.CV_HOUGH_GRADIENT, dp=15,  minDist=int (radius / 1.5) + 1, minRadius=int(radius) - 10, maxRadius= int(radius) + 10, param1= 100, param2= int(2 * 3.14 * radius - 10))
circles = np.uint16(np.around(circles))

for r in circles[0,:]:
    cv2.circle(img,(r[0],r[1]),r[2],(255,255,255),2)

print len(circles[0])
cv2.imshow('detected circles',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
