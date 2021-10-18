from src_hthick import *
import cv2 as cv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', default = 'images/bg100.jpg', help='Name of Input File (default: images/bg100.jpeg)')
    args=parser.parse_args()
    
    ellipse = ellipsePoints([50,150],[25,100])
    test = np.zeros((101,301),np.uint8)
    print(ellipse)
    for point in ellipse:
        print(point)
        test.itemset(point, 255)
    

    img = cv.imread(args.input)
    if img is None:
        sys.exit("Could not read the image.")
    
    cv.imshow("Display Window", test)
    k = cv.waitKey(0)


    return 0

if __name__ == '__main__':
    main()
