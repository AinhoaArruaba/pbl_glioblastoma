from utils import img_utils

import numpy as np
from skimage import morphology
from skimage import measure
from skimage import filters
from skimage import exposure

from medpy.filter import smoothing

from scipy import ndimage


def __contrast_enhancement(raw_slices):
    enhanced_slices = np.array(
        [exposure.equalize_adapthist(s) for s in raw_slices])
    return enhanced_slices


# Skull stripping algorithm implementation for 3D brain MRI based in McStrip algorithm


def __get_brain_coarse_mask(raw_slices):
    coarse_masks = []
    # Thresholding with Otsu's method
    threshold = filters.threshold_multiotsu(raw_slices, classes=5)
    # Apply threshold to obtain binary slice
    for index in enumerate(raw_slices):
        coarse_masks.append(raw_slices[index[0], :, :] > threshold[0])

    return np.array(coarse_masks)


def __get_brain_thresh_mask(raw_slices, coarse_masks):
    thresh_masks = []
    # Thresholding with Multi-Otsu's method
    lv1_filtered_slices = apply_mask(raw_slices, coarse_masks)
    thresholds = filters.threshold_multiotsu(lv1_filtered_slices, classes=5)

    coarse_masks = morphology.erosion(coarse_masks, morphology.ball(1))

    all_labels = measure.label(coarse_masks)
    regions = measure.regionprops(all_labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    region_masks = np.ndarray(
        [raw_slices.shape[0], raw_slices.shape[1], raw_slices.shape[2]], dtype=np.int8)
    region_masks[::] = 0
    region_masks = region_masks + \
        np.where(all_labels == regions[0].label, 1, 0)

    region_masks = morphology.dilation(region_masks, morphology.ball(1))
    lv1_filtered_slices = apply_mask(raw_slices, region_masks)

    # Apply threshold to obtain binary slice
    for index in enumerate(lv1_filtered_slices):
        thresh_masks.append(np.logical_and(lv1_filtered_slices[index[0], :, :] > thresholds[1],
                                           lv1_filtered_slices[index[0], :, :] < thresholds[3]))

    thresh_masks = np.array(thresh_masks)
    thresh_masks = morphology.erosion(thresh_masks, morphology.ball(2))

    all_labels = measure.label(thresh_masks)
    regions = measure.regionprops(all_labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    thresh_masks = np.ndarray(
        [raw_slices.shape[0], raw_slices.shape[1], raw_slices.shape[2]], dtype=np.int8)
    thresh_masks[::] = 0
    thresh_masks = thresh_masks + \
        np.where(all_labels == regions[0].label, 1, 0)

    thresh_masks = morphology.dilation(thresh_masks, morphology.ball(3))
    for index in range(thresh_masks.shape[0]):
        thresh_masks[index, :, :] = ndimage.binary_fill_holes(
            thresh_masks[index, :, :])
    thresh_masks = morphology.closing(thresh_masks, morphology.ball(4))
    for index in range(thresh_masks.shape[0]):
        thresh_masks[index, :, :] = ndimage.binary_fill_holes(
            thresh_masks[index, :, :])

    return thresh_masks


def __get_brain_bse_mask(raw_slices, coarse_masks, thresh_mask):
    # http://brainsuite.org/processing/surfaceextraction/bse/
    # Apply anisotropic difussion filter
    lv2_filtered_slices = apply_mask(raw_slices, coarse_masks)

    slices_filtered_anisotropic = np.array(smoothing.anisotropic_diffusion(
        lv2_filtered_slices))
    # Apply Marr-Hildreth filtering algorithm (laplacian of gaussian filter)
    slices_filtered_log = ndimage.gaussian_laplace(
        slices_filtered_anisotropic, 0.5)

    slices_binary = slices_filtered_log < 0
    slices_eroded = morphology.erosion(slices_binary, morphology.ball(6))
    all_labels = measure.label(slices_eroded)
    regions = measure.regionprops(all_labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    bse_masks = np.ndarray(
        [raw_slices.shape[0], raw_slices.shape[1], raw_slices.shape[2]], dtype=np.int8)
    bse_masks[::] = 0
    region_masks = bse_masks + np.where(all_labels == regions[0].label, 1, 0)
    region_masks_dilated = morphology.dilation(
        region_masks, morphology.ball(2))

    bse_masks = ndimage.binary_fill_holes(region_masks_dilated)
    for _ in range(4):
        bse_masks = morphology.dilation(bse_masks, morphology.cube(1))
        bse_masks = ndimage.binary_fill_holes(bse_masks)
        bse_masks = morphology.closing(bse_masks, morphology.ball(1))

    for index in range(bse_masks.shape[0]):
        bse_masks[index, :, :] = ndimage.binary_fill_holes(
            bse_masks[index, :, :])

    return bse_masks


def __get_consensus_mask(coarse_masks, thresh_masks, bse_masks):
    consensus_mask = coarse_masks
    for index in range(coarse_masks.shape[0]):
        for row in range(coarse_masks.shape[1]):
            for col in range(coarse_masks.shape[2]):
                if (coarse_masks[index, row, col] and bse_masks[index, row, col]) or (thresh_masks[index, row, col] and bse_masks[index, row, col]) or (thresh_masks[index, row, col] and coarse_masks[index, row, col]):
                    consensus_mask[index, row, col] = True
                else:
                    consensus_mask[index, row, col] = False

    return consensus_mask


def __skull_strip_mcstrip_algorithm(raw_slices, display=False):
    coarse_masks = __get_brain_coarse_mask(raw_slices)
    thresh_masks = __get_brain_thresh_mask(raw_slices, coarse_masks)
    bse_masks = __get_brain_bse_mask(raw_slices, coarse_masks, thresh_masks)
    consensus_mask = __get_consensus_mask(
        coarse_masks, thresh_masks, bse_masks)

    if display:
        img_utils.plot_stack('Coarse mask', coarse_masks)
        img_utils.plot_stack('Thresh mask', thresh_masks)
        img_utils.plot_stack('BSE mask', bse_masks)
        img_utils.plot_stack('Consensus mask', consensus_mask)

    return consensus_mask


def apply_mask(raw_slices, mask):
    masked_slices = []
    for index in range(raw_slices.shape[0]):
        masked_slices.append(mask[index, :, :]*raw_slices[index, :, :])

    return np.array(masked_slices)


def image_preprocessing(raw_slices, dicom_data, display):
    enhanced_slices = __contrast_enhancement(raw_slices)

    no_skull_mask = __skull_strip_mcstrip_algorithm(enhanced_slices, display)
    no_skull_slices = apply_mask(enhanced_slices, no_skull_mask)
    return no_skull_mask, no_skull_slices


def image_preprocessing_brainweb(raw_slices, display):
    no_skull_mask = __skull_strip_mcstrip_algorithm(raw_slices, display)
    no_skull_slices = apply_mask(raw_slices, no_skull_mask)

    return no_skull_mask, no_skull_slices
