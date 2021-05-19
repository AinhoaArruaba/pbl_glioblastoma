"""
app.py

"""
# Developed modules
import dicom_file_handler as dicom_handler
import segmentation as seg
import img_processing_functions as img_func
import utils.img_utils as img_utils
import numpy as np

# Python libraries
import os
import warnings

warnings.filterwarnings("ignore")

dataset_stand = 'dataset_standarized'
dataset_no_skull = 'dataset_no_skull'
img_types = ['t1', 't1c', 't2', 'flair']

if not os.path.exists(dataset_stand):
    print("MRI standarization started")
    # Read database data
    database_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "database")
    # Obtain subject list
    subjects = dicom_handler.get_subject_list(database_path)

    mri_slices_dicom = {}
    mri_slices_stand = {}
    raw_slices = {}
    subject_folders = {}

    for s in subjects:
        print("Reading subject " + s)
        if not '.ds_store' in s.lower():
            scan_folder = dicom_handler.identify_scan_folder(
                os.path.join(database_path, s))

            mri_slices_dicom[s] = {}
            raw_slices[s] = {}
            subject_folders[s] = {}
            for folder in scan_folder[1]:
                if folder in img_types:
                    print("MRI type -> " + folder)
                    subject_folder = os.path.join(scan_folder[0], folder)
                    mri_slices_dicom[s][folder] = dicom_handler.load_subject_scan(
                        os.path.join(scan_folder[0], folder))

                    raw_slices[s][folder] = dicom_handler.extract_raw_img_array(
                        mri_slices_dicom[s][folder])
                    subject_folders[s][folder] = subject_folder

    # Apply MRI standarization
    print("Standarizing...")
    mri_slices_stand = img_func.mri_standarization(
        raw_slices, subjects, 'landmarks.txt', False)

    os.mkdir(dataset_stand)

    print("Saving standarized MRI images")
    for s in subjects:
        os.mkdir(os.path.join(dataset_stand, s))
        for mri_type in subject_folders[s]:
            dicom_handler.save_subject_scan(
                mri_slices_stand[s][mri_type], s, subject_folders[s][mri_type], mri_type, dataset_stand)


print("MRI skull-stripping started")
mri_slices_dicom = {}
raw_slices = {}
mri_slices_noskull = {}
subject_folders = {}

# Obtain subject list
database_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), dataset_stand)
subjects = dicom_handler.get_subject_list(database_path)

# Extract skull
for s in subjects:
    print("Reading subject " + s)
    if not s.lower() == '.ds_store':
        mri_slices_dicom[s] = {}
        raw_slices[s] = {}
        subject_folders[s] = {}
        mri_slices_noskull[s] = {}

        for mri_type in os.listdir(os.path.join(database_path, s)):
            if mri_type in img_types:
                if not os.path.exists(os.path.join(dataset_no_skull, s, mri_type)):
                    print("Skull-stripping for MRI type " + mri_type.upper())
                    mri_slices_dicom[s][mri_type] = dicom_handler.load_subject_scan(
                        os.path.join(database_path, s, mri_type))
                    raw_slices[s][mri_type] = dicom_handler.extract_raw_img_array(
                        mri_slices_dicom[s][mri_type])
                    mri_slices_noskull[s][mri_type] = img_func.image_skull_strip(
                        raw_slices[s][mri_type], False)
                    img_utils.plot_stack(s + " " + mri_type, np.array(
                        mri_slices_noskull[s][mri_type][1]), block=False)
                    subject_folders[s][mri_type] = os.path.join(
                        database_path, s, mri_type)

if not os.path.exists(dataset_no_skull):
    os.mkdir(dataset_no_skull)
database_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), dataset_no_skull)

print("Saving skull-stripped MRI images")
for s in subjects:
    if not os.path.exists(os.path.join(dataset_no_skull, s)):
        os.mkdir(os.path.join(dataset_no_skull, s))
    for mri_type in subject_folders[s]:
        dicom_handler.save_subject_scan(
            mri_slices_noskull[s][mri_type][1], s, subject_folders[s][mri_type], mri_type, dataset_no_skull)

# mri_slices_segmented[s] = seg.segmentation(
#     mri_slices_stand[s], False)
# img_utils.plot_stack(
#     s, np.array(mri_slices_segmented[s]), rows=6, cols=4, start_with=0)


print('Done')
