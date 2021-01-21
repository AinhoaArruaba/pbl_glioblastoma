import os
import time
import subprocess
import nibabel as nib

import img_processing_functions as img_func
from utils import img_utils


def wait_file_created(path, time_wait=10):
    time_counter = 0
    while not os.path.exists(path):
        time.sleep(1)
        time_counter += 1
        if time_counter > time_wait:
            break

    return not time_counter > time_wait


def remove_files(file_list):
    for f in file_list:
        if os.path.exists(f):
            os.remove(f)


# For each subject in eval_database
#   Load brain volume
#   Load available mask
#   Compute mask with implemented mcstrip algorithm
#   Compute selected metrics and save
# Create CSV file with obtained metric values
# Generate comparation figure with 3 subjects ??


# Compute Jaccard similarity
