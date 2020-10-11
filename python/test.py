import imutils
import cv2
import numpy as np

def main():
  image = cv2.imread("images/2-4-100x.jpg")
  
  cv2.imshow("2-4-100x", image)
  cv2.waitKey(0)
