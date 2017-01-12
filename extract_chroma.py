import numpy as np
import skimage.color as color
import skimage.io as io

def extract_a_channel(image_path):
	img_rgb = io.imread(image_path)
	img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
	img_l = img_lab[:,:,1] # pull out A channel	
	return img_l


def extract_b_channel(image_path):
	img_rgb = io.imread(image_path)
	img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
	img_l = img_lab[:,:,2] # pull out B channel	
	return img_l

def extract_l_channel(image_path):
	img_rgb = io.imread(image_path)
	img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
	img_l = img_lab[:,:,0] # pull out L channel	
	return img_l