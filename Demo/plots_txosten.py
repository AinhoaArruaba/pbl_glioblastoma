import os
import numpy as np
import nibabel as nib
import utils.img_utils as img_utils
import img_processing_functions as img_func
import evaluation_script

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
