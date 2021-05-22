import os
import numpy as np
import dicom_file_handler as dicom_handler
import utils.img_utils as img_utils

dataset_no_skull = 'dataset_no_skull'
database_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), dataset_no_skull)
subjects = dicom_handler.get_subject_list(database_path)

img_types = ['t1', 't1c', 't2', 'flair']
subjects.sort()
for s in subjects:
    for mri_type in os.listdir(os.path.join(database_path, s)):
        if mri_type in img_types:
            print("Skull-stripping for subject " +
                  s + " MRI type " + mri_type.upper())
            mri_slices_dicom = dicom_handler.load_subject_scan(
                os.path.join(database_path, s, mri_type))
            raw_slices = dicom_handler.extract_raw_img_array(mri_slices_dicom)
            img_utils.plot_stack(s + " " + mri_type, raw_slices)
