from os import path
import sys
import numpy as np
import cv2 as cv


def get_features(file_name):
    obj_original = cv.imread(path.join(file_name), cv.IMREAD_COLOR)

    # error check
    if obj_original is None:
    	print 'ERROR:Couldn\'t find the object image with the provided path.'
    	sys.exit()

    # basic feature detection works in grayscale
    obj = cv.cvtColor(obj_original, cv.COLOR_BGR2GRAY)

    try:
    	sift = cv.SIFT()
    except:
    	print("ERROR: SIFT not found! cv2 version: ",cv.__version__)

    obj_keypoints, obj_descriptors = sift.detectAndCompute(obj,None)

    return obj_descriptors






