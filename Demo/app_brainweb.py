import os
import nibabel as nib
import utils.img_utils as img_utils
import img_processing_functions as img_func

image_data_t1 = nib.load(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'brainweb', 't1_icbm_normal_1mm_pn3_rf20.mnc'))
image_data_t2 = nib.load(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'brainweb', 't2_icbm_normal_1mm_pn3_rf20.mnc'))
image_data_pd = nib.load(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'brainweb', 'pd_icbm_normal_1mm_pn3_rf20.mnc'))

raw_img_t1 = image_data_t1.get_fdata()
raw_img_t2 = image_data_t2.get_fdata()
raw_img_pd = image_data_pd.get_fdata()
# mask_t1, no_skull_img_t1 = img_func.image_preprocessing_brainweb(
#     raw_img_t1, False)
# img_utils.plot_stack('No skull T1w', no_skull_img_t1)

# mask_t2, no_skull_img_t2 = img_func.image_preprocessing_brainweb(
#     raw_img_t2, False)
# img_utils.plot_stack('No skull T2w', no_skull_img_t2)
mask_pd, no_skull_img_pd = img_func.image_preprocessing_brainweb(
    raw_img_pd, False)
img_utils.plot_stack('No skull PD', no_skull_img_pd)
# img_func.image_preprocessing_brainweb(raw_img_pd, True)

# new_image = nib.Nifti1Image(raw_img_t1, image_data_t1.affine)
# nib.save(new_image, os.path.join('brainweb', 'brainweb_t1.nii'))
# new_image = nib.Nifti1Image(raw_img_t2, image_data_t2.affine)
# nib.save(new_image, os.path.join('brainweb', 'brainweb_t2.nii'))
# new_image = nib.Nifti1Image(raw_img_pd, image_data_pd.affine)
# nib.save(new_image, os.path.join('brainweb', 'brainweb_pd.nii'))

print("Done")
