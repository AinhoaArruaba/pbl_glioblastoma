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
dataset_segmentation = 'dataset_segmentation'
dataset_segmentation_masks = 'dataset_segmentation_masks'
dataset_segmentation_masks_validated = 'dataset_segmentation_masks_validated'
dataset_segmentation_contours = 'dataset_segmentation_contours'
dataset_features = 'dataset_features'
img_types = ['t1', 't1c', 't2', 'flair']

##############################################################
#################### STANDARDIZE DATASET #####################
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

##############################################################
####################### EXTRACT SKULL ########################
if not os.path.exists(dataset_no_skull):
    mri_slices_dicom = {}
    raw_slices = {}
    mri_slices_noskull = {}
    subject_folders = {}

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

##############################################################
################# SEGMENT TUMOR AND EDEMA ####################
if not os.path.exists(dataset_segmentation):
    mri_slices_dicom = {}
    raw_slices = {}
    mri_slices_tumor_segmented = {}
    mri_slices_tumor_masks = {}
    mri_slices_tumor_masks_validated = {}
    subject_folders = {}
    m_t_A_t2 = {}

    # Obtain subject list
    database_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), dataset_no_skull)
    subjects = dicom_handler.get_subject_list(database_path)

    # Segmentation
    for s in subjects:
        mri_slices_dicom[s] = {}
        raw_slices[s] = {}
        subject_folders[s] = {}
        mri_slices_tumor_segmented[s] = {}
        mri_slices_tumor_masks[s] = {}
        mri_slices_tumor_masks_validated[s] = {}
        m_t_A_t2[s] = []

        # Get mri types and reverse it to analyse t2 before flair
        found_mri_types = os.listdir(os.path.join(database_path, s))
        found_mri_types = found_mri_types[::-1]

        for mri_type in found_mri_types:
            mri_slices_dicom[s][mri_type] = dicom_handler.load_subject_scan(
                os.path.join(database_path, s, mri_type))
            raw_slices[s][mri_type] = dicom_handler.extract_raw_img_array(
                mri_slices_dicom[s][mri_type])

            #if mri_type == 't1c' or mri_type == 't1':
             #   mri_slices_tumor_segmented[s][mri_type], mri_slices_tumor_masks[s][mri_type], mri_slices_tumor_masks_validated[s][mri_type] = seg.get_tumor_segmentation(
              #      raw_slices[s][mri_type], False)

            if mri_type == 't2':
                #m_t_A_t2[s] = seg.get_t2_mask(
                 #   raw_slices[s][mri_type], False)
                    mri_slices_tumor_segmented[s][mri_type], mri_slices_tumor_masks[s][mri_type], mri_slices_tumor_masks_validated[s][mri_type] = seg.get_tumor_segmentation(
                        raw_slices[s][mri_type], False)

            if mri_type == 'flair':
                #mri_slices_tumor_segmented[s][mri_type], mri_slices_tumor_masks[s][mri_type] = seg.get_edema_segmentation(
                 #   raw_slices[s][mri_type], m_t_A_t2[s], False)
                mri_slices_tumor_segmented[s][mri_type], mri_slices_tumor_masks[s][mri_type], mri_slices_tumor_masks_validated[s][mri_type] = seg.get_tumor_segmentation(
                    raw_slices[s][mri_type], False)

            subject_folders[s][mri_type] = os.path.join(
                database_path, s, mri_type)

    if not os.path.exists(dataset_segmentation):
        os.mkdir(dataset_segmentation)

    if not os.path.exists(dataset_segmentation_masks):
        os.mkdir(dataset_segmentation_masks)

    if not os.path.exists(dataset_segmentation_masks_validated):
        os.mkdir(dataset_segmentation_masks_validated)

    # Save segmentations
    for s in subjects:
        if not os.path.exists(os.path.join(dataset_segmentation, s)):
            os.mkdir(os.path.join(dataset_segmentation, s))
        for mri_type in subject_folders[s]:
            dicom_handler.save_subject_scan_seg(
                mri_slices_tumor_segmented[s][mri_type], s, subject_folders[s][mri_type], mri_type, dataset_segmentation)

        if not os.path.exists(os.path.join(dataset_segmentation_masks, s)):
            os.mkdir(os.path.join(dataset_segmentation_masks, s))
        for mri_type in subject_folders[s]:
            dicom_handler.save_subject_scan_seg(
                mri_slices_tumor_masks[s][mri_type], s, subject_folders[s][mri_type], mri_type, dataset_segmentation_masks)

        if not os.path.exists(os.path.join(dataset_segmentation_masks_validated, s)):
            os.mkdir(os.path.join(dataset_segmentation_masks_validated, s))
        for mri_type in subject_folders[s]:
            dicom_handler.save_subject_scan_seg(
                mri_slices_tumor_masks_validated[s][mri_type], s, subject_folders[s][mri_type], mri_type, dataset_segmentation_masks_validated)

##############################################################
###################### VALIDATE MASKS ########################
if not os.path.exists(dataset_segmentation_masks_validated):
    mri_slices_dicom = {}
    raw_slices = {}
    mri_slices_tumor_masks = {}
    raw_slices_masks = {}
    mri_slices_tumor_masks_validated = {}
    mri_slices_segmented = {}
    subject_folders = {}

    # Obtain subject list
    database_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), dataset_no_skull)
    masks_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), dataset_segmentation_masks)
    subjects = dicom_handler.get_subject_list(masks_path)

    # Validation
    for s in subjects:
        mri_slices_dicom[s] = {}
        raw_slices[s] = {}
        subject_folders[s] = {}
        mri_slices_tumor_masks[s] = {}
        raw_slices_masks[s] = {}
        mri_slices_tumor_masks_validated[s] = {}
        mri_slices_segmented[s] = {}

        found_mri_types = os.listdir(os.path.join(database_path, s))
        found_mri_types = found_mri_types[::-1]

        for mri_type in found_mri_types:
            mri_slices_dicom[s][mri_type] = dicom_handler.load_subject_scan(
                os.path.join(database_path, s, mri_type))
            raw_slices[s][mri_type] = dicom_handler.extract_raw_img_array(
                mri_slices_dicom[s][mri_type])

            mri_slices_tumor_masks[s][mri_type] = dicom_handler.load_subject_scan(
                os.path.join(masks_path, s, mri_type))
            raw_slices_masks[s][mri_type] = dicom_handler.extract_raw_img_array(
                mri_slices_tumor_masks[s][mri_type])

            mri_slices_tumor_masks_validated[s][mri_type] = seg.validate_masks(raw_slices[s][mri_type], raw_slices_masks[s][mri_type])
            mri_slices_segmented[s][mri_type] = [seg*val for seg,val in zip(
                raw_slices_masks[s][mri_type], mri_slices_tumor_masks_validated[s][mri_type])]

            subject_folders[s][mri_type] = os.path.join(
                database_path, s, mri_type)

    if not os.path.exists(dataset_segmentation_masks_validated):
        os.mkdir(dataset_segmentation_masks_validated)
    if not os.path.exists(dataset_segmentation):
        os.mkdir(dataset_segmentation)

    # Save segmentations
    for s in subjects:
        if not os.path.exists(os.path.join(dataset_segmentation_masks_validated, s)):
            os.mkdir(os.path.join(dataset_segmentation_masks_validated, s))
        for mri_type in subject_folders[s]:
            dicom_handler.save_subject_scan_seg(
                mri_slices_tumor_masks_validated[s][mri_type], s, subject_folders[s][mri_type], mri_type, dataset_segmentation_masks_validated)

        if not os.path.exists(os.path.join(dataset_segmentation, s)):
            os.mkdir(os.path.join(dataset_segmentation, s))
        for mri_type in subject_folders[s]:
            dicom_handler.save_subject_scan_seg(
                mri_slices_segmented[s][mri_type], s, subject_folders[s][mri_type], mri_type, dataset_segmentation)

##############################################################
################ CREATE AND SAVE CONTOURS ####################
if not os.path.exists(dataset_segmentation_contours):
    mri_slices_dicom = {}
    raw_slices = {}
    raw_slices_masks = {}
    mri_slices_dicom_masks = {}
    mri_slices_contour = {}
    subject_folders = {}

    # Obtain subject list
    database_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), dataset_no_skull)

    database_masks = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), dataset_segmentation_masks)
    subjects = dicom_handler.get_subject_list(database_masks)

    # Create directory
    if not os.path.exists(dataset_segmentation_contours):
        os.mkdir(dataset_segmentation_contours)

    # Contour drawing
    for s in subjects:
        mri_slices_dicom[s] = {}
        raw_slices[s] = {}
        raw_slices_masks[s] = {}
        mri_slices_dicom_masks[s] = {}
        mri_slices_contour[s] = {}

        found_mri_types = os.listdir(os.path.join(database_masks, s))
        found_mri_types = found_mri_types[::-1]

        for mri_type in found_mri_types:
            mri_slices_dicom[s][mri_type] = dicom_handler.load_subject_scan(
                os.path.join(database_path, s, mri_type))
            raw_slices[s][mri_type] = dicom_handler.extract_raw_img_array(
                mri_slices_dicom[s][mri_type])

            mri_slices_dicom_masks[s][mri_type] = dicom_handler.load_subject_scan(
                os.path.join(database_masks, s, mri_type))
            raw_slices_masks[s][mri_type] = dicom_handler.extract_raw_img_array(
                mri_slices_dicom_masks[s][mri_type])

            mri_slices_contour[s][mri_type] = seg.save_tumor_contours(
                os.path.join(dataset_segmentation_contours, 'mask_' + s + '_' + mri_type + '.png'),
                s, raw_slices[s][mri_type], raw_slices_masks[s][mri_type])

##############################################################
################ CALCULATE FEATURES ####################
if not os.path.exists(dataset_features):
    mri_slices_dicom = {}
    raw_slices = {}
    raw_slices_masks = {}
    mri_slices_dicom_masks = {}
    mri_slices_contour = {}
    subject_folders = {}

    # Obtain subject list
    database_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), dataset_segmentation)

    database_masks = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), dataset_segmentation_masks_validated)
    subjects = dicom_handler.get_subject_list(database_masks)

    # Create directory
    if not os.path.exists(dataset_features):
        os.mkdir(dataset_features)

    # Contour drawing
    for s in subjects:
        mri_slices_dicom[s] = {}
        raw_slices[s] = {}
        raw_slices_masks[s] = {}
        mri_slices_dicom_masks[s] = {}
        mri_slices_contour[s] = {}

        found_mri_types = os.listdir(os.path.join(database_masks, s))
        found_mri_types = found_mri_types[::-1]

        for mri_type in found_mri_types:
            mri_slices_dicom[s][mri_type] = dicom_handler.load_subject_scan(
                os.path.join(database_path, s, mri_type))
            raw_slices[s][mri_type] = dicom_handler.extract_raw_img_array(
                mri_slices_dicom[s][mri_type])

            mri_slices_dicom_masks[s][mri_type] = dicom_handler.load_subject_scan(
                os.path.join(database_masks, s, mri_type))
            raw_slices_masks[s][mri_type] = dicom_handler.extract_raw_img_array(
                mri_slices_dicom_masks[s][mri_type])

            #if mri_type == 't1c' or mri_type == 't1':
                #seg.extract_features(raw_slices[s][mri_type], raw_slices_masks[s][mri_type])

            #if mri_type == 't2':
                #seg.extract_features(raw_slices[s][mri_type], raw_slices_masks[s][mri_type])

            #if mri_type == 'flair':
                #seg.extract_features(raw_slices[s][mri_type], raw_slices_masks[s][mri_type])

            subject_folders[s][mri_type] = os.path.join(
                database_path, s, mri_type)

print('Done')
