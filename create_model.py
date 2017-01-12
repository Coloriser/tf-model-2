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
		feature = load_from_pickle(path)
		brisk_features.append(feature)
	return brisk_features

def load_a_channel_chroma(paths):
	a_channel_chromas = []
	for path in paths:
		chroma = load_from_pickle(path)
		a_channel_chromas.append(chroma)
	return a_channel_chromas

def load_b_channel_chroma(paths):
	b_channel_chromas = []
	for path in paths:
		chroma = load_from_pickle(path)
		b_channel_chromas.append(chroma)
	return b_channel_chromas

def make_model(x, y):

	print("X :",x.shape)
	print("Y :",y.shape)

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

	# Training
	model = tflearn.DNN(network, tensorboard_verbose=2)
	model.fit({'input': x}, {'target': y} , n_epoch=1)

	return model


def make_a_model():
	brisk_paths = load_brisk_paths()
	a_channel_paths = load_a_channel_chroma_paths()

	print("loading brisk features...")
	brisk_features = load_brisk_features(brisk_paths)

	print("modifying the shape of input and output")
	train_x = np.array(brisk_features).reshape([1, 344, 64, 1])

	print("loading a channel chroma...")
	a_channel_chromas = load_a_channel_chroma(a_channel_paths)

	train_y_a_channel = np.array(a_channel_chromas).reshape(1,54810)
	train_y_a_channel = train_y_a_channel+128
	train_y_a_channel = train_y_a_channel/256.0


	print("Generating A channel model")
	model_a_channel = make_model(train_x, train_y_a_channel)
	model_a_channel.save("model/a_channel.model")

def make_b_model():
	brisk_paths = load_brisk_paths()

	print("loading brisk features...")
	brisk_features = load_brisk_features(brisk_paths)

	print("modifying the shape of input and output")
	train_x = np.array(brisk_features).reshape([1, 344, 64, 1])


	b_channel_paths = load_b_channel_chroma_paths()

	print("loading b channel chroma...")
	b_channel_chromas = load_b_channel_chroma(b_channel_paths)

	train_y_b_channel = np.array(b_channel_chromas).reshape(1,54810)
	train_y_b_channel = train_y_b_channel+128
	train_y_b_channel = train_y_b_channel/256.0


	print("Generating B channel model")
	model_b_channel = make_model(train_x, train_y_b_channel)
	model_b_channel.save("model/b_channel.model")


def main():
	args = parse_arguments()
	print(args)
	if args.a:
		print("Training model based on a-channel")
		make_a_model()
	if args.b:	
		print("Training model based on b-channel")
		make_b_model()
	if not args.a and not args.b:
		print("ERROR: use -h for HELP")

main()
