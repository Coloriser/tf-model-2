from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import numpy as np
import pickle
import argparse

def parse_arguments():						#argument parser -d for the pathlist
    parser = argparse.ArgumentParser(description='Trains the model, to be used after running pre-works')
    parser.add_argument('-a', help='to train based on a-channel', required=False,action="store_true", default=False)
    parser.add_argument('-b', help='to train based on b-channel', required=False,action="store_true", default=False)
    args = parser.parse_args()
    return args

def normalize_brisk_array(brisk_features):		#to normalize the shape of each numpy array in brisk array
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

def load_from_pickle(path):
	f = open(path, "rb")
	value = pickle.load(f)
	f.close()
	return value

def load_brisk_paths():
	return load_from_pickle('brisk_paths')

def load_a_channel_chroma_paths():
	return load_from_pickle('a_channel_chroma_paths')

def load_b_channel_chroma_paths():
	return load_from_pickle('b_channel_chroma_paths')

def load_brisk_features(paths):
	brisk_features = []
	for path in paths:
		try:
			feature = load_from_pickle(path)
			brisk_features.append(feature)
		except:
			print("Error at Brisk load")	
	return brisk_features

def load_a_channel_chroma(paths):
	a_channel_chromas = []
	for path in paths:
		try:				
			chroma = load_from_pickle(path)
			a_channel_chromas.append(chroma)
		except:
			print("Error at A_Channel load")	
	return a_channel_chromas

def load_b_channel_chroma(paths):
	b_channel_chromas = []
	for path in paths:
		try:
			chroma = load_from_pickle(path)
			b_channel_chromas.append(chroma)
		except:
			print("Error at B_Channel load")	
	return b_channel_chromas

def pickle_shape( x, y):
	a = {"input_shape" : x.shape,"output_shape" : y.shape}
	path = 'shape_of_in_and_out'
	f = open(path, "wb")
	value = pickle.dump(a, f)
	f.close()
	return value		

def make_model(x, y):

	print("X :",x.shape)
	print("Y :",y.shape)


	# Building convolutional networ
	network = input_data(shape=[None, x.shape[1], x.shape[2], 1], name='input')

	network = fully_connected(network, 128, activation='sigmoid')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='sigmoid')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='sigmoid')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='sigmoid')
	network = dropout(network, 0.8)

	network = fully_connected(network, y.shape[1], activation='sigmoid')
	network = regression(network, optimizer='adam', learning_rate=0.01,
	                     loss='categorical_crossentropy', name='target')

	# Training
	model = tflearn.DNN(network, tensorboard_verbose=2)
	model.fit({'input': x}, {'target': y} , n_epoch=100)

	return model


def make_a_model():
	brisk_paths = load_brisk_paths()
	a_channel_paths = load_a_channel_chroma_paths()

	print("loading brisk features...")
	brisk_features = load_brisk_features(brisk_paths)

	# print("Before normalization")
	# print(brisk_features[0].shape)
	# print(brisk_features[1].shape)

	print("Normalizing Brisk features")
	modified_brisk_features = normalize_brisk_array(brisk_features)

	No_Of_Test_Items = len(modified_brisk_features)
	
	# print("After normalization")
	# print(modified_brisk_features[0].shape)
	# print(modified_brisk_features[1].shape)

	print("modifying the shape of input and output")
	train_x = np.array(modified_brisk_features).reshape([No_Of_Test_Items, modified_brisk_features[0].shape[0], modified_brisk_features[0].shape[1], 1])

	print("loading a channel chroma...")
	a_channel_chromas = load_a_channel_chroma(a_channel_paths)

	train_y_a_channel = np.array(a_channel_chromas).reshape(No_Of_Test_Items,-1)
	train_y_a_channel = train_y_a_channel+128
	train_y_a_channel = train_y_a_channel/256.0

	print("Pickling shapes")
	pickle_shape(train_x,train_y_a_channel)

	print("Generating A channel model")
	model_a_channel = make_model(train_x, train_y_a_channel)
	model_a_channel.save("model/a_channel.model")

def make_b_model():
	brisk_paths = load_brisk_paths()

	print("loading brisk features...")
	brisk_features = load_brisk_features(brisk_paths)

	# print("Before normalization")
	# print(brisk_features[0].shape)
	# print(brisk_features[1].shape)

	print("Normalizing Brisk features")
	modified_brisk_features = normalize_brisk_array(brisk_features)
	No_Of_Test_Items = len(modified_brisk_features)


	# print("After normalization")
	# print(modified_brisk_features[0].shape)
	# print(modified_brisk_features[1].shape)

	print("modifying the shape of input and output")
	train_x = np.array(modified_brisk_features).reshape([No_Of_Test_Items, modified_brisk_features[0].shape[0], modified_brisk_features[0].shape[1], 1])

	print("train_x: ",train_x.shape)


	b_channel_paths = load_b_channel_chroma_paths()

	print("loading b channel chroma...")
	b_channel_chromas = load_b_channel_chroma(b_channel_paths)

	train_y_b_channel = np.array(b_channel_chromas).reshape(No_Of_Test_Items, -1)
	train_y_b_channel = train_y_b_channel+128
	train_y_b_channel = train_y_b_channel/256.0

	print("Pickling shapes")
	pickle_shape(train_x,train_y_b_channel)

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
