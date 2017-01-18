import subprocess

change_resolution = convert *.jpg -resize 100x100! new

rename_files = "for FN in *; do mv $FN $FN.jpg ; done"