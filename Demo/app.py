"""
app.py

"""
# Developed modules
import dicom_file_handler as dicom_handler
import segmentation
import img_processing_functions as img_func
# Python libraries
import os


# Read database data
database_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "database")
# Obtain subject list
subjects = dicom_handler.get_subject_list(database_path)

mri_slices_dicom = {}
mri_slices_preprocessed = {}
for s in subjects:
    scan_folder = dicom_handler.identify_scan_folder(
        os.path.join(database_path, s))

    for folder in scan_folder[1]:
        subject_folder = s + '_' + folder
        mri_slices_dicom[subject_folder] = dicom_handler.load_subject_scan(
            os.path.join(scan_folder[0], folder))

        raw_img = dicom_handler.extract_raw_img_array(
            mri_slices_dicom[subject_folder])

        mri_slices_preprocessed[s] = img_func.image_preprocessing(
            raw_img, mri_slices_dicom, False)

        # dicom_handler.write_dicom(
        #     mri_slices_preprocessed[s], mri_slices_dicom[subject_folder], folder, s)

print('Done')
