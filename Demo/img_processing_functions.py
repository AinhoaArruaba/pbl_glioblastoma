from utils import img_utils

import numpy as np
from skimage import morphology
from skimage import measure
from skimage import filters
from skimage import exposure

from medpy.filter import smoothing

from scipy import ndimage


def contrast_enhancement(raw_slices):
    enhanced_slices = np.array(
        [exposure.equalize_adapthist(s) for s in raw_slices])
    return enhanced_slices


# Skull stripping algorithm implementation for 3D brain MRI based in McStrip algorithm


def get_brain_coarse_mask(raw_slices, display=False):
    coarse_masks = []
    # Thresholding with Otsu's method
    threshold = filters.threshold_mean(raw_slices)
    # Apply threshold to obtain binary slice
    for index in enumerate(raw_slices):
        coarse_masks.append(raw_slices[index[0], :, :] > threshold)
    coarse_masks = ndimage.binary_fill_holes(coarse_masks)

    if display:
        img_utils.plot_stack('', np.array(coarse_masks))

    return np.array(coarse_masks)


def get_brain_thresh_mask(raw_slices, coarse_masks, display=False):
    thresh_masks = []
    # lv1_filtered_slices = np.array([raw_slices[index[0], :, :] *
    #                                 coarse_masks[index[0], :, :] for index in enumerate(raw_slices)])
    # Thresholding with Multi-Otsu's method
    thresholds = filters.threshold_multiotsu(raw_slices, classes=4)
    # Apply threshold to obtain binary slice
    for index in enumerate(raw_slices):
        thresh_masks.append(np.logical_and(np.array(raw_slices[index[0], :, :] < thresholds[2]), np.array(
            raw_slices[index[0], :, :] > thresholds[0])))

    thresh_masks = ndimage.binary_fill_holes(thresh_masks)

    if display:
        # img_utils.plot_hist_thresholds(raw_slices[14, :, :], thresholds)
        # img_utils.plot_stack('', np.array(lv1_filtered_slices))
        img_utils.plot_stack('Thresh mask', np.array(thresh_masks))

    return np.array(thresh_masks)


def get_brain_bse_mask(raw_slices, coarse_masks, thresh_mask, display=False):
    # http://brainsuite.org/processing/surfaceextraction/bse/
    # Apply anisotropic difussion filter
    lv2_filtered_slices = np.array(
        [raw_slices[index[0], :, :] * thresh_mask[index[0], :, :] for index in enumerate(raw_slices)])
    slices_filtered_anisotropic = np.array(smoothing.anisotropic_diffusion(
        lv2_filtered_slices))
    # Apply Marr-Hildreth filtering algorithm (laplacian of gaussian filter)
    slices_filtered_log = ndimage.gaussian_laplace(
        slices_filtered_anisotropic, 0.5)

    slices_binary = slices_filtered_log < 0
    slices_eroded = morphology.erosion(slices_binary, morphology.cube(7))
    all_labels = measure.label(slices_eroded)
    regions = measure.regionprops(all_labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    bse_masks = np.ndarray(
        [raw_slices.shape[0], raw_slices.shape[1], raw_slices.shape[2]], dtype=np.int8)
    bse_masks[::] = 0
    region_masks = bse_masks + np.where(all_labels == regions[0].label, 1, 0)
    region_masks_dilated = morphology.dilation(
        region_masks, morphology.cube(3))

    for index in range(region_masks_dilated.shape[0]):
        bse_masks[index, :, :] = ndimage.binary_fill_holes(
            region_masks_dilated[index, :, :])

    for _ in range(4):
        bse_masks = morphology.dilation(bse_masks, morphology.cube(2))
        bse_masks = ndimage.binary_fill_holes(bse_masks)
        bse_masks = morphology.closing(bse_masks, morphology.ball(4))
        bse_masks = morphology.erosion(bse_masks, morphology.cube(1))

    for index in range(bse_masks.shape[0]):
        bse_masks[index, :, :] = ndimage.binary_fill_holes(
            bse_masks[index, :, :])

    if display:
        img_utils.plot_stack('BSE mask', bse_masks)

    return bse_masks


def skull_strip_mcstrip_algorithm(raw_slices, display=False):
    coarse_masks = get_brain_coarse_mask(raw_slices, display=False)
    thresh_masks = get_brain_thresh_mask(
        raw_slices, coarse_masks, display=False)
    bse_masks = get_brain_bse_mask(
        raw_slices, coarse_masks, thresh_masks, display=False)

    no_skull_slices = []
    for index in range(raw_slices.shape[0]):
        no_skull_slices.append(
            bse_masks[index, :, :]*raw_slices[index, :, :])

    if display:
        img_utils.plot_stack('Masked brain', np.array(no_skull_slices))

    return no_skull_slices


def image_preprocessing(raw_slices, dicom_data, display):
    enhanced_slices = contrast_enhancement(raw_slices)

    no_skull_slices = skull_strip_mcstrip_algorithm(enhanced_slices, display)

    return no_skull_slices


def image_preprocessing_brainweb(raw_slices, display):
    no_skull_slices = skull_strip_mcstrip_algorithm(raw_slices, display)

    return no_skull_slices
