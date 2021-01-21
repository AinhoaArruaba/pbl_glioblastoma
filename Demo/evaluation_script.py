import os
import time
import subprocess
import nibabel as nib

import img_processing_functions as img_func
import utils.img_utils as img_utils


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

# LOAD
# raw_img_t1 = image_data_t1.get_fdata()
# pixel_dims = (raw_img_t1.shape[0], raw_img_t1.shape[1],
#               raw_img_t1.shape[2])
# raw_images = np.zeros(pixel_dims, dtype=np.int8)
# for index_1 in range(pixel_dims[0]):
#     for index_2 in range(pixel_dims[1]):
#         for index_3 in range(pixel_dims[2]):
#             raw_images[index_1, index_2, index_3] = raw_img_t1[index_1, index_2,
#                                                                index_3][0] if raw_img_t1[index_1, index_2, index_3][0] >= 0 else 0


# Compute Jaccard similarity
