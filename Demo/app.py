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


# Read database data
database_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "database")
# Obtain subject list
subjects = dicom_handler.get_subject_list(database_path)

mri_slices_dicom = {}
mri_slices_stand = {}
mri_slices_segmented = {}
raw_slices = {}
subject_folders = []

for s in subjects:
    scan_folder = dicom_handler.identify_scan_folder(
        os.path.join(database_path, s))

    for folder in scan_folder[1]:
        subject_folder = s + '_' + folder
        mri_slices_dicom[subject_folder] = dicom_handler.load_subject_scan(
            os.path.join(scan_folder[0], folder))

        raw_slices[subject_folder] = dicom_handler.extract_raw_img_array(
            mri_slices_dicom[subject_folder])
        subject_folders.append(subject_folder)

# Apply MRI standarization
mri_slices_stand = img_func.mri_standarization(
    raw_slices, subject_folders, True)

# Extract segmentations
for s in subjects:
    pass
    # img_utils.plot_stack(
    #     s, mri_slices_stand[s][1], rows=6, cols=4, start_with=0)

    # mri_slices_segmented[s] = seg.segmentation(
    #     mri_slices_stand[s], False)

    # img_utils.plot_stack(
    #     s, np.array(mri_slices_segmented[s]), rows=6, cols=4, start_with=0)


print('Done')
