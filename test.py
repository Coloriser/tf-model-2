from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import numpy as np
import pickle
import skimage.color as color
import skimage.io as io

import argparse

import pre_works_test as pre_works

x=[]
y=[]

def parse_arguments():                      #argument parser -d for the pathlist
    parser = argparse.ArgumentParser(description='Tests the model, to be used after creating the model\n To run in ab mode, run in -a and -b first')
    parser.add_argument("--path",'-p', help='the path to input (default: ./dataset/test)', required=False, default="./dataset/test")
    parser.add_argument('-a', help='to train based on a-channel', required=False,action="store_true", default=False)
    parser.add_argument('-b', help='to train based on b-channel', required=False,action="store_true", default=False)
    parser.add_argument('-ab', help='to train based on a&b-channel', required=False,action="store_true", default=False)
    args = parser.parse_args()
    return args    


def load_from_pickle(path):
    f = open(path, "rb")
    value = pickle.load(f)
    f.close()
    return value

def save_blob(content, path):
    f = open(path, "wb")
    pickle.dump(content, f)
    f.close()    

def load_brisk_paths():
    return load_from_pickle('paths_for_test/brisk_paths')

def load_luminance_paths():
    return load_from_pickle('paths_for_test/l_channel_luminance_paths')

def load_brisk_features(paths):
    brisk_features = []
    for path in paths:
        feature = load_from_pickle(path)
        brisk_features.append(feature)
    return brisk_features

def load_luminance(paths):
    luminance = []
    for path in paths:
        feature = load_from_pickle(path)
        luminance.append(feature)
    return luminance

def load_b_channel_chroma_paths():
    return load_from_pickle('paths_for_test/b_channel_chroma_paths')

def load_b_channel_chroma(paths):
    b_channel_chromas = []
    for path in paths:
        chroma = load_from_pickle(path)
        b_channel_chromas.append(chroma)
    return b_channel_chromas

def load_a_channel_chroma_paths():
    return load_from_pickle('paths_for_test/a_channel_chroma_paths')

def load_a_channel_chroma(paths):
    a_channel_chromas = []
    for path in paths:
        chroma = load_from_pickle(path)
        a_channel_chromas.append(chroma)
    return a_channel_chromas


def import_shape_from_pickle( ):
    path = 'shape_of_in_and_out'
    # f = open(path, "rb")
    # value = pickle.load(f)
    # f.close()
    value = load_from_pickle(path)
    print("x", value["input_shape"], "y", value["output_shape"])
    return value["input_shape"], value["output_shape"]  

def load_model(path):
    global x,y
    #importing shapes from file
    x, y = import_shape_from_pickle( )

    # Building convolutional network
    network = input_data(shape=[None, x[1], x[2], 1], name='input')

    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)

    network = fully_connected(network, y[1], activation='sigmoid')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')



    model = tflearn.DNN(network)
    model.load(path)
    return model



def reconstruct(l_arr,a_arr,b_arr, count, type):

    print(l_arr.shape)
    print(a_arr.shape)
    print(b_arr.shape)
    
    img = np.vstack(([l_arr.T], [a_arr.T], [b_arr.T])).T
    rgb_image = color.lab2rgb(img)
    io.imsave("predicted_images/"+str(count)+"_" + type + ".jpg", rgb_image)

def scale_image(chroma):
    chroma = np.array(chroma)
    chroma = chroma*256
    chroma = chroma - 128
    chroma = np.reshape(chroma, (200,200))
    return chroma

def normalize_brisk_array(brisk_features):      #to normalize the shape of each numpy array in brisk array
    maximum_shape = (0,0)
    modified_brisk_features=[]

    # to find the maximum_shape
    for each_feature in brisk_features:
        if(each_feature.shape > maximum_shape):
            maximum_shape = each_feature.shape
    # to normalize brisk feature shape
    for each_feature in brisk_features:
            y = each_feature.copy()
            y.resize(maximum_shape)
            modified_brisk_features.append(y)
    return modified_brisk_features

def load_a_model():
    print("Loading 'a' model")
    return load_model("model/a_channel.model")

def load_b_model():
    print("Loading 'b' model")
    return load_model("model/b_channel.model")

def main():

    args = parse_arguments()

    if not (args.a or args.b or args.ab): #Check if only one case is true
        print("ERROR: use -h for HELP")
        exit()

    DATASET_PATH = args.path

    print("processing images")
    pre_works.process_images(DATASET_PATH)

    ab_flag = False    

    # print(args)
    if args.a:
        print("Testing model based on a-channel")
    if args.b:  
        print("Testing model based on b-channel")
    if args.ab:  
        print("Testing model based on a&b-channel")
        ab_flag = True    

    print("Loading paths")
    brisk_paths = load_brisk_paths()
    luminance_paths = load_luminance_paths()
    if args.a:
        b_channel_paths = load_b_channel_chroma_paths()
        print("loading b channel chroma...")
        b_channel_chromas = load_b_channel_chroma(b_channel_paths)
    if args.b:    
        a_channel_paths = load_a_channel_chroma_paths()
        print("loading a channel chroma...")
        a_channel_chromas = load_a_channel_chroma(a_channel_paths)

    print("loading brisk features...")
    brisk_features = load_brisk_features(brisk_paths)

    print("loading luminance...")
    luminance = load_luminance(luminance_paths)

    print("Normalizing Brisk features")
    modified_brisk_features = normalize_brisk_array(brisk_features)

    No_Of_Test_Items = len(modified_brisk_features)

    print("modifying the shape of input and output")
    train_x = np.array(modified_brisk_features).reshape([No_Of_Test_Items, modified_brisk_features[0].shape[0], modified_brisk_features[0].shape[1], 1])

    if args.a:
        model_a_channel = load_a_model()
        predictions = model_a_channel.predict(train_x)
        print("Dumping a_chroma")
        save_blob(predictions, 'predicted_a_chroma')
        for i in range(len(predictions)):
            a_channel_chroma = scale_image(predictions[i])
            reconstruct(luminance[i], a_channel_chroma,b_channel_chromas[i], i, 'A')
    if args.b:
        model_b_channel = load_b_model()
        predictions = model_b_channel.predict(train_x)
        print("Dumping b_chroma")
        save_blob(predictions, 'predicted_b_chroma')    
        for i in range(len(predictions)):
            b_channel_chroma = scale_image(predictions[i])
            reconstruct(luminance[i], a_channel_chromas[i],b_channel_chroma, i, 'B')

    if args.ab:
        print("Loading a_chroma")
        predictions_A = load_from_pickle("predicted_a_chroma")
        print("Loading b_chroma")
        predictions_B = load_from_pickle("predicted_b_chroma")  
        for i in range(len(predictions_A)):
            a_channel_chroma = scale_image(predictions_A[i])
            b_channel_chroma = scale_image(predictions_B[i])
            reconstruct(luminance[i], a_channel_chroma,b_channel_chroma, i, 'AB')


main()        