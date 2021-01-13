"""
app.py

"""
# Developed modules
import dicom_file_reader as dicom_reader
import segmentation
import img_processing_functions as img_func
# Python libraries
import os

# Read database data
database_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "database")
# Obtain subject list
subjects = dicom_reader.get_subject_list(database_path)

mri_slices_dicom = {}
mri_slices_preprocessed = {}
for s in subjects:
    scan_folder = dicom_reader.identify_scan_folder(
        os.path.join(database_path, s))

    mri_slices_dicom[s] = dicom_reader.load_subject_scan(
        os.path.join(scan_folder[0], scan_folder[1][0]))

    img = dicom_reader.extract_raw_img_array(mri_slices_dicom[s])
    # img_func.plot_hist(s, img)
    # img_func.plot_stack(s, img)
    mri_slices_preprocessed[s] = img_func.image_preprocessing(img, True)

print(subjects)
