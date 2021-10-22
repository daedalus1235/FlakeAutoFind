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

    img = cv.imread(args.input)
    if img is None:
        sys.exit("Could not read the image.")
    
    dim = img.shape[0:2]

    n = 3

    flat=[None, None]

    b,g,r = cv.split(img)

    ar = dim[0]/dim[1]
    a = [0]*n
    c = (1400,900)

    flatten(r, flat, a, c, ar, n)

    cv.imshow("Display Window", flat[0])
    k = cv.waitKey(0)


    return 0

if __name__ == '__main__':
    main()
