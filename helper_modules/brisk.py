from os import path
import sys
import numpy as np
import cv2 as cv

def get_features(file_name):

    # alternative detectors, descriptors, matchers, parameters ==> different results

    # Object Features
    # obj_original = cv.imread(path.join(file_name),cv.CV2_LOAD_IMAGE_COLOR)
    obj_original = cv.imread(path.join(file_name), cv.IMREAD_COLOR)

    #error checking
    if obj_original is None:
        print 'Couldn\'t find the object image with the provided path.'
        sys.exit()

    # basic feature detection works in grayscale
    obj = cv.cvtColor(obj_original, cv.COLOR_BGR2GRAY)

    #for cv version 3.1.0 
    if cv.__version__ == '3.1.0':
        brisk = cv.BRISK_create()
        (obj_keypoints, obj_descriptors) = brisk.detectAndCompute(obj, None)
        
        #dump numpy array to file
        obj_descriptors.dump(resultname)

        return obj_descriptors
        
    #for cv versions 2.x.x
    detector = cv.BRISK(thresh=10, octaves=1)
    extractor = cv.DescriptorExtractor_create('BRISK')  # non-patented. Thank you!

    matcher = cv.BFMatcher(cv.NORM_L2SQR)

    # keypoints are "interesting" points in an image:
    obj_keypoints = detector.detect(obj, None)


    # this lines up each keypoint with a mathematical description
    obj_keypoints, obj_descriptors = extractor.compute(obj, obj_keypoints)

    return obj_descriptors