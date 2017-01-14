import brisk
from glob import glob
from os.path import exists, join, basename, splitext
import numpy
from extract_chroma import extract_a_channel,extract_b_channel,extract_l_channel
import pickle

EXTENSIONS = [".jpg",".png"]

def get_image_paths(path="dataset/train"):
    """Get the list of all the image files in the train directory"""
    image_paths = []
    image_paths.extend([join(path, basename(fname))
                    for fname in glob(path + "/*")
                    if splitext(fname)[-1].lower() in EXTENSIONS])
    return image_paths


def get_brisk_features(image_path):
	""" Process an image and return the BRISK features"""
	return brisk.get_features(image_path)	

def get_a_channel_chroma(image_path):
    """ Process an image and return the A channel chroma"""
    return extract_a_channel(image_path)
    
def get_b_channel_chroma(image_path):
    """ Process an image and return the B channel chroma"""
    return extract_b_channel(image_path)
    
def get_l_channel_luminance(image_path):
    """ Process an image and return the L channel luminance"""
    return extract_l_channel(image_path)


def create_brisk_path(image_path):
    """ Create path to store the BRISK feature given path to an image"""
    path =  image_path.split(".")[0]
    path = path.replace("dataset", "brisk_features")
    return path + ".brisk"

def create_a_channel_chroma_path(image_path):
    """ Create path to store the A channel chroma given path to an image"""
    path =  image_path.split(".")[0]
    path = path.replace("dataset", "a_channel_chroma")
    path = path.replace("/train", "")
    return path + ".a_channel_chroma"

def create_b_channel_chroma_path(image_path):
    """ Create path to store the A channel chroma given path to an image"""
    path =  image_path.split(".")[0]
    path = path.replace("dataset", "b_channel_chroma")
    path = path.replace("/train", "")
    return path + ".b_channel_chroma"

def create_l_channel_luminance_path(image_path):
    """ Create path to store the A channel chroma given path to an image"""
    path =  image_path.split(".")[0]
    path = path.replace("dataset", "l_channel_luminance")
    path = path.replace("/train", "")
    return path + ".l_channel_luminance"


def save_blob(content, path):
    f = open(path, "wb")
    pickle.dump(content, f)
    f.close()


def main():

    image_paths = get_image_paths()
    brisk_paths = map(create_brisk_path, image_paths)
    a_channel_chroma_paths = map(create_a_channel_chroma_path, image_paths)
    b_channel_chroma_paths = map(create_b_channel_chroma_path, image_paths)
    l_channel_luminance_paths = map(create_l_channel_luminance_path, image_paths)

    for i in range(len(image_paths)):

        brisk_features = get_brisk_features(image_paths[i])
        a_channel_chroma = get_a_channel_chroma(image_paths[i])
        b_channel_chroma = get_b_channel_chroma(image_paths[i])
        l_channel_luminance = get_l_channel_luminance(image_paths[i])

        save_blob(brisk_features, brisk_paths[i])
        save_blob(a_channel_chroma, a_channel_chroma_paths[i])
        save_blob(b_channel_chroma, b_channel_chroma_paths[i])
        save_blob(l_channel_luminance, l_channel_luminance_paths[i])

    save_blob(brisk_paths, "brisk_paths")
    save_blob(a_channel_chroma_paths, "a_channel_chroma_paths")
    save_blob(b_channel_chroma_paths, "b_channel_chroma_paths")
    save_blob(l_channel_luminance_paths, "l_channel_luminance_paths")

main()

