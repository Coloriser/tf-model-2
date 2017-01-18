import numpy as np
import skimage.color as color
import skimage.io as io
import pickle



def real_a_channel(a_arr):

	b_arr = np.zeros( shape = a_arr.shape)
	l_arr = np.zeros( shape = a_arr.shape)

	img = np.vstack(([l_arr.T], [a_arr.T], [b_arr.T])).T
	rgb_image = color.lab2rgb(img)

	io.imsave("real_a_channel.jpg", rgb_image)


def predicted_a_channel(a_arr):

	b_arr = np.zeros( shape = a_arr.shape )
	l_arr = np.zeros( shape = a_arr.shape )

	img = np.vstack(([l_arr.T], [a_arr.T], [b_arr.T])).T
	rgb_image = color.lab2rgb(img)

	io.imsave("predicted_a_channel.jpg", rgb_image)


file_name = 'small_col.jpg'
img_rgb = io.imread(file_name)	
img_lab = color.rgb2lab(img_rgb)

# l_arr = img_lab[:,:,0]
# print("l_arr.shape : ",l_arr.shape)


f = open('predicted.chroma', 'r')
predicted_a_arr = pickle.load(f)
f.close()

real_a_arr = img_lab[:,:,1]

real_a_channel(real_a_arr)
predicted_a_channel(predicted_a_arr)