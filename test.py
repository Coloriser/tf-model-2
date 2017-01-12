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

def load_from_pickle(path):
    f = open(path, "rb")
    value = pickle.load(f)
    f.close()
    return value


def load_brisk_paths():
    return load_from_pickle('brisk_paths')

def load_luminance_paths():
    return load_from_pickle('l_channel_luminance_paths')

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
    return load_from_pickle('b_channel_chroma_paths')

def load_b_channel_chroma(paths):
    b_channel_chromas = []
    for path in paths:
        chroma = load_from_pickle(path)
        b_channel_chromas.append(chroma)
    return b_channel_chromas


def load_model(path):

    # Building convolutional network
    network = input_data(shape=[None, 344, 64, 1], name='input')

    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, 0.8)

    network = fully_connected(network, 54810, activation='sigmoid')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')



    model = tflearn.DNN(network)
    model.load(path)
    return model



def reconstruct(l_arr,a_arr,b_arr):

    print(l_arr.shape)
    print(a_arr.shape)
    print(b_arr.shape)

    img = np.vstack(([l_arr.T], [a_arr.T], [b_arr.T])).T
    rgb_image = color.lab2rgb(img)
    io.imsave("reconstructed.jpg", rgb_image)



def main():
    model_a_channel = load_model("model/a_channel.model")

    brisk_paths = load_brisk_paths()
    luminance_paths = load_luminance_paths()
    b_channel_paths = load_b_channel_chroma_paths()

    print("loading brisk features...")
    brisk_features = load_brisk_features(brisk_paths)

    print("loading luminance...")
    luminance = load_luminance(luminance_paths)

    print("loading b channel chroma...")
    b_channel_chromas = load_b_channel_chroma(b_channel_paths)


    print("modifying the shape of input and output")
    train_x = np.array(brisk_features).reshape([1, 344, 64, 1])

    prediction = model_a_channel.predict(train_x)

    a_channel_chroma = prediction[0]
    a_channel_chroma = np.array(a_channel_chroma)
    a_channel_chroma = a_channel_chroma*256
    a_channel_chroma = a_channel_chroma - 128
    a_channel_chroma = np.reshape(a_channel_chroma, (203,270))

    reconstruct(luminance[0], a_channel_chroma, b_channel_chromas[0])


main()