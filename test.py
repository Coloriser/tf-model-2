from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import numpy as np
import pickle
import argparse

import sys
sys.path.insert(0, './helper_modules')
import helper_functions as hf

# import pre_works_test as pre_works

# x=[]
# y=[]

def parse_arguments():                      #argument parser -d for the pathlist
    parser = argparse.ArgumentParser(description='Tests the model, to be used after creating the model. To run in ab mode, run in -a and -b first')
    parser.add_argument("--path",'-p', help='the path to input (default: ./dataset/test)', required=False, default="./dataset/test")
    parser.add_argument('-a', help='to train based on a-channel', required=False,action="store_true", default=False)
    parser.add_argument('-b', help='to train based on b-channel', required=False,action="store_true", default=False)
    parser.add_argument('-ab', help='to train based on a&b-channel', required=False,action="store_true", default=False)
    args = parser.parse_args()
    return args    


def load_a_model():
    print("Loading 'a' model")
    return load_model("model/a_channel.model")

def load_b_model():
    print("Loading 'b' model")
    return load_model("model/b_channel.model")

def load_model(path):
    # global x,y
    #importing shapes from file
    x, y = hf.import_shape_from_pickle( )

    # Building convolutional network
    network = input_data(shape=[None, x[1], x[2], 1], name='input')
    print(network)

    #1
    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)
    #2
    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)
    #3
    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)
    #4
    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)
    #5
    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)
    #6
    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)
    #7
    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)
    #8
    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)
    #9
    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)
    #10
    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)

    network = fully_connected(network, y[1], activation='sigmoid')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='mean_square', name='target')



    model = tflearn.DNN(network)
    model.load(path)
    return model


def prereq_load_and_compute( SIFT=False):
    print("Loading feature paths")
    if SIFT==True:
        print("SIFT")
        paths = hf.load_sift_paths('test')

    else:
        print("BRISK")
        paths = hf.load_brisk_paths('test')
    print("loading features...")
    features = hf.load_features(paths)
    print(str(len(features)) + " items loaded.")    
    print("Normalizing features")
    modified_feature_arr = hf.normalize_array(features, mode = 'test')
    No_Of_Test_Items = len(modified_feature_arr)    


    print("modifying the shape of input and output")
    test_x = np.array(modified_feature_arr).reshape([No_Of_Test_Items, modified_feature_arr[0].shape[0], modified_feature_arr[0].shape[1], 1])

    print("test_x shape: ",test_x.shape)

    return test_x

def predict_and_dump(test_x, mode):
    
    luminance_paths = hf.load_luminance_paths('test')
    print("loading luminance...")
    luminance = hf.load_luminance(luminance_paths)

    if mode=='a':
        b_channel_paths = hf.load_b_channel_chroma_paths('test')
        print("loading b channel chroma...")
        b_channel_chromas = hf.load_b_channel_chroma(b_channel_paths)
        # try:
        model = load_a_model()
        # except:
            # print("Error loading model")

    if mode=='b':    
        a_channel_paths = hf.load_a_channel_chroma_paths('test')
        print("loading a channel chroma...")
        a_channel_chromas = hf.load_a_channel_chroma(a_channel_paths)
        # try:
        model = load_b_model()
        # except:
            # print("Error loading model")

    predictions = model.predict(test_x)
    print("Dumping predictions")
    if(mode == 'a'):
        hf.save_blob(predictions, 'predicted_a_chroma')
        for i in range(len(predictions)):
            a_channel_chroma = hf.scale_image(predictions[i])
            hf.reconstruct(luminance[i], a_channel_chroma,b_channel_chromas[i], i, 'A')
    if(mode == 'b'):
        hf.save_blob(predictions, 'predicted_b_chroma')    
        for i in range(len(predictions)):
            b_channel_chroma = hf.scale_image(predictions[i])
            hf.reconstruct(luminance[i], a_channel_chromas[i],b_channel_chroma, i, 'B')
    

def main():

    args = parse_arguments()
    DATASET_PATH = args.path
    mode = 'NONE'
    if not (args.a or args.b or args.ab): #Check if only one case is true
        print("ERROR: use -h for HELP")
        exit()
        return
    # print(args)
    if args.a:
        print("Testing model based on a-channel")
        mode = 'a'
    if args.b:  
        print("Testing model based on b-channel")
        mode = 'b'
    if args.ab:  
        print("Testing model based on a&b-channel")
        mode = 'ab'

    
    # FOR AKHEEL
    # print("processing images")
    # pre_works.process_images(DATASET_PATH)

    if(mode != 'ab') :   
        test_x = prereq_load_and_compute( SIFT = True )    
        predict_and_dump(test_x, mode)
        return

    # IF AB MODE
    print("Loading a_chroma")
    predictions_A = hf.load_from_pickle("predicted_a_chroma")
    print("Loading b_chroma")
    predictions_B = hf.load_from_pickle("predicted_b_chroma")  
    luminance_paths = hf.load_luminance_paths('test')
    print("loading luminance...")
    luminance = hf.load_luminance(luminance_paths)
    for i in range(len(predictions_A)):
        a_channel_chroma = hf.scale_image(predictions_A[i])
        b_channel_chroma = hf.scale_image(predictions_B[i])
        hf.reconstruct(luminance[i], a_channel_chroma,b_channel_chroma, i, 'AB')


main()        