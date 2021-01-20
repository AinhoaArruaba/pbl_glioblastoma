import os
import nibabel as nib
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
img_func.image_preprocessing_brainweb(raw_img_t1, True)
img_func.image_preprocessing_brainweb(raw_img_t2, True)
img_func.image_preprocessing_brainweb(raw_img_pd, True)

print("Done")
