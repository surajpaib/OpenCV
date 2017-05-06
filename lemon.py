import cv2
import cv2.cv as cv
import numpy as np
import timeit


class LemonRecognizer(object):
    def __init__(self, image_file):
        self.image = cv2.imread(image_file)
        self.circles = None
        self.radius = None
        self.feature_image = None

    def get_image(self):
        return self.image

    def get_min_enclosed_circle(self):
        """
        Use contours to find minimum enclosing circle to get an estimate for running hough transform
        :return: Minimum circle radius
        """
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        ## Extract Hue Value from HSV representation ( Taking lemon hue as a feature extractor )
        im = hsv[:, :, 0]
        ## Blur image to remove noise
        im = cv2.blur(im, (3, 3))
        im = cv2.bitwise_not(im)
        ## Use Otsu's method for automatic thresholding to convert to b/w image
        ret, bw = cv2.threshold(im, 0 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.blur(bw, (4, 4))

        ## Start finding contour using opencv contour library
        contour, hierarchy = cv2.findContours(bw, cv.CV_RETR_TREE, cv.CV_CHAIN_APPROX_SIMPLE)
        cnt = contour[0]
        (x, y), self.radius = cv2.minEnclosingCircle(cnt)
        self.feature_image = bw
        return (x, y), self.radius

    def hough_circle_transform(self):
        """
        Use hough circle method to find circles in the image to fit a given criteria ( In this case, 
        to find circles close to the minimum enclosed circle from contour estimation
        :return: 
        """
        r_eff = int(self.radius)
        circles = cv2.HoughCircles(self.feature_image, method=cv.CV_HOUGH_GRADIENT, dp=5, minDist=int(r_eff * 0.5 + 1),
                                   minRadius=int(r_eff - r_eff * 0.4), maxRadius=int(r_eff + r_eff * 0.009),
                                   param1=10, param2=int(2 * 3.14 * int(r_eff - r_eff * 0.57)))
        self.circles = circles[0, :]
        return self.circles


