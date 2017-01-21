import SIFT_module as sift
import numpy
from extract_chroma import extract_a_channel,extract_b_channel,extract_l_channel
import pickle

def create_sift_path(image_path):
    """ Create path to store the SIFT feature given path to an image"""
    path =  image_path.split(".")[-2]
    filename =  path.split("/")[-1]
    path = "sift_features/test/" + filename + ".sift"
    return path

def create_a_channel_chroma_path(image_path):
    """ Create path to store the A channel chroma given path to an image"""
    path =  image_path.split(".")[-2]
    filename =  path.split("/")[-1]
    path = "a_channel_chroma/test/" + filename + ".a_channel_chroma"
    # path = path.replace("dataset", "a_channel_chroma")
    # path = path.replace("/train", "")
    return path

def create_b_channel_chroma_path(image_path):
    """ Create path to store the A channel chroma given path to an image"""
    path =  image_path.split(".")[-2]
    filename =  path.split("/")[-1]
    path = "b_channel_chroma/test/" + filename + ".b_channel_chroma"
    # path = path.replace("dataset", "b_channel_chroma")
    # path = path.replace("/train", "")
    return path

def create_l_channel_luminance_path(image_path):
    """ Create path to store the A channel chroma given path to an image"""
    path =  image_path.split(".")[-2]
    filename =  path.split("/")[-1]
    path = "l_channel_luminance/test/" + filename + ".l_channel_luminance"
    # path = path.replace("dataset", "l_channel_luminance")
    # path = path.replace("/train", "")
    return path


def save_blob(content, path):
    f = open(path, "wb")
    pickle.dump(content, f)
    f.close()


def get_sift_features(image_path):
	""" Process an image and return the SIFT features"""
	return sift.get_features(image_path)	

def get_a_channel_chroma(image_path):
    """ Process an image and return the A channel chroma"""
    return extract_a_channel(image_path)
    
def get_b_channel_chroma(image_path):
    """ Process an image and return the B channel chroma"""
    return extract_b_channel(image_path)
    
def get_l_channel_luminance(image_path):
    """ Process an image and return the L channel luminance"""
    return extract_l_channel(image_path)


def process_images(image_paths, thread_no):

    sift_paths = map(create_sift_path, image_paths)
    a_channel_chroma_paths = map(create_a_channel_chroma_path, image_paths)
    b_channel_chroma_paths = map(create_b_channel_chroma_path, image_paths)
    l_channel_luminance_paths = map(create_l_channel_luminance_path, image_paths)

    print ("Paths generated for thread " + str(thread_no))

    for i in range(len(image_paths)):
        print("Thread " + str(thread_no) + " working on " + str(i+1) + " out of " + str(len(image_paths)) + ' : ' + str(image_paths[i]))
        try:
            sift_features = get_sift_features(image_paths[i])
            a_channel_chroma = get_a_channel_chroma(image_paths[i])
            b_channel_chroma = get_b_channel_chroma(image_paths[i])
            l_channel_luminance = get_l_channel_luminance(image_paths[i])

        except:
            print("Error")

        else:
            save_blob(sift_features, sift_paths[i])
            save_blob(a_channel_chroma, a_channel_chroma_paths[i])
            save_blob(b_channel_chroma, b_channel_chroma_paths[i])
            save_blob(l_channel_luminance, l_channel_luminance_paths[i])
    return    

def save_paths( image_paths ):
    sift_paths = map(create_sift_path, image_paths)
    a_channel_chroma_paths = map(create_a_channel_chroma_path, image_paths)
    b_channel_chroma_paths = map(create_b_channel_chroma_path, image_paths)
    l_channel_luminance_paths = map(create_l_channel_luminance_path, image_paths)

    save_blob(sift_paths, "paths_for_test/sift_paths")
    save_blob(a_channel_chroma_paths, "paths_for_test/a_channel_chroma_paths")
    save_blob(b_channel_chroma_paths, "paths_for_test/b_channel_chroma_paths")
    save_blob(l_channel_luminance_paths, "paths_for_test/l_channel_luminance_paths")
    return

# process_images(".g/dataset/test")