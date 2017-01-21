import pickle



def normalize_array(feature_arr):		#to normalize the shape of each numpy array in brisk array
	maximum_shape = (0,0)
	modified_array=[]

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

def load_from_pickle(path):
	f = open(path, "rb")
	value = pickle.load(f)
	f.close()
	return value

def load_brisk_paths():
	return load_from_pickle('brisk_paths')

def load_sift_paths():
	return load_from_pickle('sift_paths')

def load_a_channel_chroma_paths():
	return load_from_pickle('a_channel_chroma_paths')

def load_b_channel_chroma_paths():
	return load_from_pickle('b_channel_chroma_paths')


def load_features(paths):
	features = []
	print len(paths)
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



