"""
Lemon.py
"""
import timeit
import numpy as np
import cv2
import cv2.cv as cv


class LemonRecognizer(object):
    """
    Class Lemon Recognizer
    """
    def __init__(self, image_file):
        self.image = cv2.imread(image_file)
        self.circles = None
        self.radius = None
        self.feature_image = None

    def get_image(self):
        """
        Function to display image
        """
        return self.image

    def convert_to_blob(self):
        """
        Use HSV values to find blobs in the images
        :return: Minimum circle radius
        """
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # Get 3 images for H, S, and V
        im1 = hsv[:, :, 0]
        im2 = hsv[:, :, 1]
        im3 = hsv[:, :, 2]
        # Use Otsu's method for automatic thresholding to convert to b/w image
        _, bw1 = cv2.threshold(im1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bw2 = cv2.threshold(im2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bw3 = cv2.threshold(im3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Combine the Saturation and Value b/w image
        concat = np.bitwise_and(bw2, bw3)
        concat = cv2.blur(concat, (5, 5))
        self.feature_image = concat
        return bw1, bw2, bw3, concat

    def blob_detecter(self):
        """
        Initializes and sets up OpenCV's Simple Blob Detector
        """
        params = cv2.SimpleBlobDetector_Params()
        # Detect circles
        params.filterByCircularity = True
        params.minCircularity = 0.15
        # Threshold for splitting images
        params.minThreshold = 200
        params.maxThreshold = 500
        # filter by color
        params.filterByColor = False
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector(params)
        # Detect blobs.
        keypoints = detector.detect(self.feature_image)
        # Draw detected blobs as circles.
        im_with_keypoints = cv2.drawKeypoints(self.image, keypoints, np.array([]), (255, 0, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Return image and keypoints
        return im_with_keypoints, keypoints
