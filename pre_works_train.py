import pre_works_train_helper as pwt
from glob import glob
from os.path import exists, join, basename, splitext

import multiprocessing as mp


EXTENSIONS = [".jpg",".png"]

def get_image_paths(path="dataset/train"):
    """Get the list of all the image files in the train directory"""
    image_paths = []
    image_paths.extend([join(path, basename(fname))
                    for fname in glob(path + "/*")
                    if splitext(fname)[-1].lower() in EXTENSIONS])
    return image_paths



def begin_threaded_execution():
    image_paths = get_image_paths()

    No_of_images = len( image_paths )
    No_of_cores = mp.cpu_count()
    images_per_core = No_of_images / No_of_cores
    threads = []

    process_list = []
    for ith_core in range(No_of_cores):
        # Building processes list
        start_point = images_per_core * ith_core
        end_point = images_per_core * (ith_core+1)

        if ith_core != No_of_cores-1:
            sub_array = image_paths[start_point:end_point]
        else:
            sub_array = image_paths[start_point:]
        print("Beginning execution of thread " + str(ith_core)  + " with " + str(len(sub_array)) + " images")
        process_list.append(mp.Process(target=pwt.process_images, args=(sub_array, ith_core)))

    for p in process_list:
        p.start()
    for p in process_list:
        p.join()    
    print("Done")
    
begin_threaded_execution()        
