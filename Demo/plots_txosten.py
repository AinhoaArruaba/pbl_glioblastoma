import os
import numpy as np
import nibabel as nib
import utils.img_utils as img_utils
import img_processing_functions as img_func
import evaluation_script
import dicom_file_handler as dicom_handler

from scipy.signal import find_peaks

dataset_stand_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "dataset_standarized")
database_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "database")

s = 'C3L-00016'
img_types = ['t1', 't1c', 't2', 'flair']
scan_folder = dicom_handler.identify_scan_folder(
    os.path.join(database_path, s))

mri_slices_dicom = {}
raw_slices = {}
for folder in scan_folder[1]:
    if folder in img_types:
        print("MRI type -> " + folder)
        subject_folder = os.path.join(scan_folder[0], folder)
        mri_slices_dicom[folder] = dicom_handler.load_subject_scan(
            os.path.join(scan_folder[0], folder))

        raw_slices[folder] = dicom_handler.extract_raw_img_array(
            mri_slices_dicom[folder])

mri_slices_dicom_stand = {}
raw_slices_stand = {}

for mri_type in os.listdir(os.path.join(dataset_stand_path, s)):
    if mri_type in img_types:
        print("MRI type -> " + mri_type)
        mri_slices_dicom_stand[mri_type] = dicom_handler.load_subject_scan(
            os.path.join(dataset_stand_path, s, mri_type))
        raw_slices_stand[mri_type] = dicom_handler.extract_raw_img_array(
            mri_slices_dicom_stand[mri_type])
n_slc = 14
figuras_plot_stand = np.zeros(
    (8, raw_slices_stand['t1'].shape[1], raw_slices_stand['t1'].shape[2]))
figuras_plot_stand[0, :, :] = raw_slices[img_types[0]][n_slc, :, :]
figuras_plot_stand[1, :, :] = raw_slices_stand[img_types[0]][n_slc, :, :]
figuras_plot_stand[2, :, :] = raw_slices[img_types[1]][n_slc, :, :]
figuras_plot_stand[3, :, :] = raw_slices_stand[img_types[1]][n_slc, :, :]
figuras_plot_stand[4, :, :] = raw_slices[img_types[2]][n_slc, :, :]
figuras_plot_stand[5, :, :] = raw_slices_stand[img_types[2]][n_slc, :, :]
figuras_plot_stand[6, :, :] = raw_slices[img_types[3]][20, :, :]
figuras_plot_stand[7, :, :] = raw_slices_stand[img_types[3]][20, :, :]

img_utils.plot_stack_documentation_img(
    'Imágenes estandarizadas', figuras_plot_stand,
    ['T1', 'T1 estandarizada', 'T1C', 'T1C estandarizada', 'T2',
        'T2 estandarizada', 'FLAIR', 'FLAIR estandarizada'],
    rows=4, cols=2)

for mri_type in img_types:
    m1 = np.min(raw_slices[mri_type])
    m2 = np.max(raw_slices[mri_type])
    [hist, edges] = np.histogram(raw_slices[mri_type], bins=m2)
    p1 = np.percentile(raw_slices[mri_type], 0)
    p2 = np.percentile(raw_slices[mri_type], 0.98)
    peaks = find_peaks(hist, prominence=1, width=3,
                       height=700, distance=3)
    img_utils.plot_hist_peaks(s, hist, peaks)

eval_database_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "database_eval", "IBSR_04")
raw_images = evaluation_script.load_subject_data(eval_database_path)
mask_t1,  _ = img_func.image_preprocessing(
    raw_images)

# Generate plots
# Lv 1
n_slc = 120
coarse_mask = mask_t1["coarse"]
coarse_masked_slices = img_func.apply_mask(raw_images, coarse_mask)
figuras_plot_lv1 = np.zeros((3, coarse_mask.shape[1], coarse_mask.shape[2]))
figuras_plot_lv1[0, :, :] = raw_images[n_slc, :, :]
figuras_plot_lv1[1, :, :] = coarse_mask[n_slc, :, :]
figuras_plot_lv1[2, :, :] = coarse_masked_slices[n_slc, :, :]

img_utils.plot_stack_documentation_img(
    'Máscara de Nivel 1', figuras_plot_lv1,
    ['Imágen original', 'Máscara', 'Resultado'],
    rows=1, cols=3)

# Lv 2
threshold_mask = mask_t1["threshold"]
thresh_masked_slices = img_func.apply_mask(raw_images, threshold_mask)
figuras_plot_lv2 = np.zeros((3, coarse_mask.shape[1], coarse_mask.shape[2]))
figuras_plot_lv2[0, :, :] = raw_images[n_slc, :, :]
figuras_plot_lv2[1, :, :] = threshold_mask[n_slc, :, :]
figuras_plot_lv2[2, :, :] = thresh_masked_slices[n_slc, :, :]

img_utils.plot_stack_documentation_img(
    'Máscara de Nivel 2', figuras_plot_lv2,
    ['Imágen original', 'Máscara', 'Resultado'],
    rows=1, cols=3)

# Lv 3
bse_mask = mask_t1["bse"]
bse_masked_slices = img_func.apply_mask(raw_images, bse_mask)
figuras_plot_lv3 = np.zeros((3, coarse_mask.shape[1], coarse_mask.shape[2]))
figuras_plot_lv3[0, :, :] = raw_images[n_slc, :, :]
figuras_plot_lv3[1, :, :] = bse_mask[n_slc, :, :]
figuras_plot_lv3[2, :, :] = bse_masked_slices[n_slc, :, :]

img_utils.plot_stack_documentation_img(
    'Máscara de Nivel 3', figuras_plot_lv3,
    ['Imágen original', 'Máscara', 'Resultado'],
    rows=1, cols=3)

# Consensus
consensus_mask = mask_t1["consensus"]
consensus_masked_slices = img_func.apply_mask(raw_images, consensus_mask)
figuras_plot_lv4 = np.zeros((3, coarse_mask.shape[1], coarse_mask.shape[2]))
figuras_plot_lv4[0, :, :] = raw_images[n_slc, :, :]
figuras_plot_lv4[1, :, :] = consensus_mask[n_slc, :, :]
figuras_plot_lv4[2, :, :] = consensus_masked_slices[n_slc, :, :]

img_utils.plot_stack_documentation_img(
    'Máscara de Consenso', figuras_plot_lv4,
    ['Imágen original', 'Máscara', 'Resultado'],
    rows=1, cols=3)
