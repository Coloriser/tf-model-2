import pickle
import numpy as np
import skimage.color as color
import skimage.io as io



def load_from_pickle(path):
	f = open(path, "rb")
	value = pickle.load(f)
	f.close()
	return value

def save_blob(content, path):
    f = open(path, "wb")
    pickle.dump(content, f)
    f.close()    	

def load_brisk_paths(mode):
	if mode=='train':
		return load_from_pickle('brisk_paths')	
	return load_from_pickle('paths_for_test/brisk_paths')

def load_sift_paths(mode):
	if mode=='train':
		return load_from_pickle('sift_paths')
	return load_from_pickle('paths_for_test/sift_paths')	
	
def load_a_channel_chroma_paths(mode):
	if mode=='train':
		return load_from_pickle('a_channel_chroma_paths')
	return load_from_pickle('paths_for_test/a_channel_chroma_paths')

def load_b_channel_chroma_paths(mode):
	if mode=='train':
		return load_from_pickle('b_channel_chroma_paths')
	return load_from_pickle('paths_for_test/b_channel_chroma_paths')


def load_luminance_paths(mode):
	if mode=='train':
		return load_from_pickle('l_channel_luminance_paths')
	return load_from_pickle('paths_for_test/l_channel_luminance_paths')

def load_features(paths):
	features = []
	# print len(paths)
	for path in paths:
		try:
			feature = load_from_pickle(path)
			features.append(feature)
		except:
			print("Error loading " + path)	
	return features

def load_a_channel_chroma(paths):
	a_channel_chromas = []
	for path in paths:
		try:				
			chroma = load_from_pickle(path)
			a_channel_chromas.append(chroma)
		except:
			print("Error loading " + path)		
	return a_channel_chromas

def load_b_channel_chroma(paths):
	b_channel_chromas = []
	for path in paths:
		try:
			chroma = load_from_pickle(path)
			b_channel_chromas.append(chroma)
		except:
			print("Error loading " + path)		
	return b_channel_chromas

def load_luminance(paths):
    luminance = []
    for path in paths:
        try:
            feature = load_from_pickle(path)
            luminance.append(feature)
        except:
            print("Error loading " + path)    
    return luminance

def pickle_shape( x, y):
	a = {"input_shape" : x.shape,"output_shape" : y.shape}
	path = 'shape_of_in_and_out'
	f = open(path, "wb")
	value = pickle.dump(a, f)
	f.close()
	return value

def import_shape_from_pickle():
    path = 'shape_of_in_and_out'
    value = load_from_pickle(path)
    if not value:
    	print("ERROR LOADING shape_of_in_and_out file")
    	exit()
    print("x", value["input_shape"], "y", value["output_shape"])
    return value["input_shape"], value["output_shape"]

def normalize_array(feature_arr, mode="train"):		#to normalize the shape of each numpy array in brisk array
	maximum_shape = (0,0)
	modified_array=[]

	if mode == 'test':
		pickled_input_shape = import_shape_from_pickle()[0]
		maximum_shape = (pickled_input_shape[1],pickled_input_shape[2])

	# to find the maximum_shape
	for each_feature in feature_arr:
		if(each_feature.shape > maximum_shape):
			maximum_shape = each_feature.shape
	# to normalize brisk feature shape
	for each_feature in feature_arr:
			y = each_feature.copy()
			y.resize(maximum_shape)
			modified_array.append(y)
	return modified_array

def reconstruct(l_arr,a_arr,b_arr, count, type):

    # print(l_arr.shape)
    # print(a_arr.shape)
    # print(b_arr.shape)
    
    img = np.vstack(([l_arr.T], [a_arr.T], [b_arr.T])).T
    rgb_image = color.lab2rgb(img)
    file_name = str(count)+"_" + type + ".jpg"
    io.imsave("predicted_images/"+file_name, rgb_image)
    print("Saved: " + file_name)

def scale_image(chroma):
    chroma = np.array(chroma)
    chroma = chroma*256
    chroma = chroma - 128
    chroma = np.reshape(chroma, (200,200))
    return chroma





