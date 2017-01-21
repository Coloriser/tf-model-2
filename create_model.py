from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import numpy as np
import argparse

import sys
sys.path.insert(0, './helper_modules')
import helper_functions as hf

def parse_arguments():						#argument parser -d for the pathlist
    parser = argparse.ArgumentParser(description='Trains the model, to be used after running pre-works')
    parser.add_argument('-a', help='to train based on a-channel', required=False,action="store_true", default=False)
    parser.add_argument('-b', help='to train based on b-channel', required=False,action="store_true", default=False)
    args = parser.parse_args()
    return args


def make_model(x, y):

	print("X :",x.shape)
	print("Y :",y.shape)


	# Building convolutional network
	network = input_data(shape=[None, x.shape[1], x.shape[2], 1], name='input')

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

	network = fully_connected(network, y.shape[1], activation='sigmoid')
	network = regression(network, optimizer='adam', learning_rate=0.01,
	                     loss='categorical_crossentropy', name='target')

	# Training
	model = tflearn.DNN(network, tensorboard_verbose=2)
	model.fit({'input': x}, {'target': y} , n_epoch=100)

	return model


def prereq_load_and_compute( mode , SIFT=False):
	if SIFT==True:
		print("SIFT")
		paths = hf.load_sift_paths('train')
	else:
		print("BRISK")
		paths = hf.load_brisk_paths('train')
	print("loading features...")
	features = hf.load_features(paths)
	print(str(len(features)) + " items loaded.")	
	print("Normalizing features")
	modified_feature_arr = hf.normalize_array(features)
	No_Of_Test_Items = len(modified_feature_arr)
	
	if mode=='a':
		a_channel_paths = hf.load_a_channel_chroma_paths('train')
		print("loading a channel chroma...")
		a_channel_chromas = hf.load_a_channel_chroma(a_channel_paths)
		print(str(len(a_channel_chromas)) + " items loaded.")	
		train_y_channel = np.array(a_channel_chromas).reshape(No_Of_Test_Items,-1)

	else:	
		b_channel_paths = hf.load_b_channel_chroma_paths('train')
		print("loading b channel chroma...")
		b_channel_chromas = hf.load_b_channel_chroma(b_channel_paths)
		print(str(len(b_channel_chromas)) + " items loaded.")
		train_y_channel = np.array(b_channel_chromas).reshape(No_Of_Test_Items, -1)

	train_y_channel = train_y_channel+128
	train_y_channel = train_y_channel/256.0

	print("modifying the shape of input and output")
	train_x = np.array(modified_feature_arr).reshape([No_Of_Test_Items, modified_feature_arr[0].shape[0], modified_feature_arr[0].shape[1], 1])
	
	print("Pickling shapes")
	hf.pickle_shape(train_x,train_y_channel)

	print("train_x shape: ",train_x.shape)
	print("train_y shape: ",train_y_channel.shape)

	return train_x, train_y_channel

def make_a_model(  ):

	train_x, train_y_a_channel = prereq_load_and_compute( mode='a' , SIFT=True)

	print("Generating A channel model")
	model_a_channel = make_model(train_x, train_y_a_channel)
	model_a_channel.save("model/a_channel.model")

def make_b_model():

	train_x, train_y_b_channel = prereq_load_and_compute( mode='b' , SIFT=True)

	print("Generating B channel model")
	model_b_channel = make_model(train_x, train_y_b_channel)
	model_b_channel.save("model/b_channel.model")


def main():
	args = parse_arguments()
	# print(args)
	if args.a:
		print("Training model based on a-channel")
		make_a_model()
	if args.b:	
		print("Training model based on b-channel")
		make_b_model()
	if not args.a and not args.b:
		print("ERROR: use -h for HELP")

main()
