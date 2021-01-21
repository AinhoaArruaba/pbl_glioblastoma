import os
import time
import subprocess
import nibabel as nib

import img_processing_functions as img_func
from utils import img_utils


def obtain_skull_segmentation_bse(raw_img, mnc_file):

    vol = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'brainweb', 'vol.nii.gz')
    if not os.path.exists(vol):
        new_image = nib.Nifti1Image(raw_img, mnc_file.affine)
        nib.save(new_image, vol)

    brain_masked_mri = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'brainweb', 'brain_masked_mri.nii.gz')

    brain_masked_mri_mask = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'brainweb', 'brain_mask.nii.gz')

    if not os.path.exists(brain_masked_mri):
        command = "echo 45194371S | sudo -S /Applications/BrainSuite19b/bin/bse -i " + \
            vol + " -o " + brain_masked_mri + " --mask " + \
            brain_masked_mri_mask + " --norotate"
        os.system(command)

        if wait_file_created(brain_masked_mri, time_wait=10):
            print("File created")
            segmented_mri_masks = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'brainweb', 'segmented_mri.nii.gz')
            segmented_mri_brain_masks = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'brainweb', 'segmented__mri_brain.nii.gz')
            command = "echo 45194371S | sudo -S /Applications/BrainSuite19b/bin/skullfinder -i " + \
                vol + " -o " + segmented_mri_masks + " -m " + \
                brain_masked_mri_mask + " -brainlabel " + segmented_mri_brain_masks
            os.system(command)

            if wait_file_created(segmented_mri_brain_masks, time_wait=30):
                vol_no_skull_mask = nib.load(segmented_mri_brain_masks)
                img_utils.plot_stack('', vol_no_skull_mask.get_fdata())

    remove_files(
        [vol, brain_masked_mri, brain_masked_mri_mask, segmented_mri_masks])


def wait_file_created(path, time_wait=10):
    time_counter = 0
    while not os.path.exists(path):
        time.sleep(1)
        time_counter += 1
        if time_counter > time_wait:
            break

    return not time_counter > time_wait


def remove_files(file_list):
    for f in file_list:
        if os.path.exists(f):
            os.remove(f)


# Delete the skull with the implemented algorithm
brainweb_t1_mnc = nib.load(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'brainweb', 't1_icbm_normal_1mm_pn3_rf20.mnc'))

raw_img_t1 = brainweb_t1_mnc.get_fdata()
# mask, brain = img_func.image_preprocessing_brainweb(raw_img_t1, False)


obtain_skull_segmentation_bse(raw_img_t1, brainweb_t1_mnc)

# Compute Jaccard similarity
