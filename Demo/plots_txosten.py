import os
import numpy as np
import nibabel as nib
import utils.img_utils as img_utils
import img_processing_functions as img_func

image_data_t1 = nib.load(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'database_eval', 'IBSR_10', 'IBSR_10_ana.nii.gz'))
raw_img_t1 = image_data_t1.get_fdata()
pixel_dims = (raw_img_t1.shape[0], raw_img_t1.shape[1],
              raw_img_t1.shape[2])
raw_images = np.zeros(pixel_dims, dtype=np.int8)
for index_1 in range(pixel_dims[0]):
    for index_2 in range(pixel_dims[1]):
        for index_3 in range(pixel_dims[2]):
            raw_images[index_1, index_2,
                       index_3] = raw_img_t1[index_1, index_2, index_3][0]


mask_t1, slices = img_func.image_preprocessing(raw_images, True)

# Generate plots
coarse_mask = mask_t1["coarse"]
coarse_masked_slices = img_func.apply_mask(raw_images, coarse_mask)
n_slc = 100
figuras_plot_lv1 = np.zeros((3, coarse_mask.shape[1], coarse_mask.shape[2]))

figuras_plot_lv1[0, :, :] = raw_images[n_slc, :, :]
figuras_plot_lv1[1, :, :] = coarse_mask[n_slc, :, :]
figuras_plot_lv1[2, :, :] = coarse_masked_slices[n_slc, :, :]

img_utils.plot_stack_documentation_img(
    'Máscara de Nivel 1', figuras_plot_lv1,
    ['Imágen original', 'Máscara', 'Resultado'],
    rows=1, cols=3)

# threshold_mask = mask_t1[2]
# thresh_masked_slices = img_func.apply_mask(raw_img_t1, threshold_mask)

# bse_mask = mask_t1[3]
# bse_masked_slices = img_func.apply_mask(raw_img_t1, bse_mask)

# consensus_mask = mask_t1[0]
# consensus_masked_slices = img_func.apply_mask(raw_img_t1, consensus_mask)
