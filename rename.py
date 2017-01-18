import subprocess

change_resolution_command = "convert ./dataset/train/*.jpg -resize 200x200! ./dataset/train/new"

rename_files_command = "for FN in ./dataset/train/*; do mv $FN ./$FN.jpg ; done"

remove_files_command = "rm -rf ./dataset/train/*.jpg"

move_files_command = "mv ./dataset/train/*.jpg ./dataset/train/old/"

rename_old_command = "mv ./dataset/train/old.* ./dataset/train/old"

path = "./dataset/train/"

print_files_command = "ls " + path 

# subprocess.call("echo Changing Directory: ", shell=True)
# subprocess.call(change_dir_commmand, shell=True)

print("Existing Structure")
subprocess.call(print_files_command, shell=True)

print("Changing resolution to 200x200")
subprocess.call(change_resolution_command, shell=True)

print("Deleting old files")
subprocess.call(move_files_command, shell=True)

print("Renaming Files")
subprocess.call(rename_files_command, shell=True)

print("New Structure")
subprocess.call(print_files_command, shell=True)

print("Renaming Old")
subprocess.call(rename_old_command, shell=True)

