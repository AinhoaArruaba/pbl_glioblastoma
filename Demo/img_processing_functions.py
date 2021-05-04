import utils.img_utils as img_utils

import numpy as np
import os
from skimage import morphology
from skimage import measure
from skimage import filters
from skimage import exposure

from medpy.filter import smoothing

from scipy import ndimage
from scipy.signal import find_peaks


def contrast_enhancement(raw_slices):
    raw_slices = img_utils.scale_range(raw_slices, 0, 1)
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

    coarse_masks_eroded = morphology.erosion(coarse_masks, morphology.ball(2))

    all_labels = measure.label(coarse_masks_eroded)
    regions = measure.regionprops(all_labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    region_masks = np.ndarray(
        [raw_slices.shape[0], raw_slices.shape[1], raw_slices.shape[2]], dtype=np.int8)
    region_masks[::] = 0
    region_masks = region_masks + \
        np.where(all_labels == regions[0].label, 1, 0)

    # region_masks = morphology.dilation(region_masks, morphology.ball(1))
    lv1_filtered_slices = apply_mask(raw_slices, region_masks)

    # Apply threshold to obtain binary slice
    for index in enumerate(lv1_filtered_slices):
        thresh_masks.append(np.logical_and(lv1_filtered_slices[index[0], :, :] > thresholds[1],
                                           lv1_filtered_slices[index[0], :, :] < thresholds[3]))

    thresh_masks = np.array(thresh_masks)
    thresh_masks = morphology.erosion(thresh_masks, morphology.ball(1))

    all_labels = measure.label(thresh_masks)
    regions = measure.regionprops(all_labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    thresh_masks = np.ndarray(
        [raw_slices.shape[0], raw_slices.shape[1], raw_slices.shape[2]], dtype=np.int8)
    thresh_masks[::] = 0
    thresh_masks = thresh_masks + \
        np.where(all_labels == regions[0].label, 1, 0)

    thresh_masks = morphology.dilation(thresh_masks, morphology.ball(1))
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
    consensus_mask = np.empty(coarse_masks.shape, dtype=bool)
    for index in range(consensus_mask.shape[0]):
        for row in range(consensus_mask.shape[1]):
            for col in range(consensus_mask.shape[2]):
                if (coarse_masks[index, row, col] and bse_masks[index, row, col]) or (thresh_masks[index, row, col] and bse_masks[index, row, col]) or (thresh_masks[index, row, col] and coarse_masks[index, row, col]):
                    consensus_mask[index, row, col] = True
                else:
                    consensus_mask[index, row, col] = False

    # Fill holes mask
    for index in range(consensus_mask.shape[0]):
        consensus_mask[index, :, :] = ndimage.binary_fill_holes(
            consensus_mask[index, :, :])

    return consensus_mask


def __skull_strip_mcstrip_algorithm(raw_slices, display=False):
    coarse_masks = __get_brain_coarse_mask(raw_slices)
    thresh_masks = __get_brain_thresh_mask(raw_slices, coarse_masks)
    bse_masks = __get_brain_bse_mask(raw_slices, coarse_masks, thresh_masks)
    consensus_mask = __get_consensus_mask(
        coarse_masks, thresh_masks, bse_masks)

    mask_dict = {"consensus": consensus_mask, "coarse": coarse_masks,
                 "threshold": thresh_masks, "bse": bse_masks}

    if display:
        img_utils.plot_stack('Coarse mask', coarse_masks)
        img_utils.plot_stack('Thresh mask', thresh_masks)
        img_utils.plot_stack('BSE mask', bse_masks)
        img_utils.plot_stack('Consensus mask', consensus_mask)

    return mask_dict


def apply_mask(raw_slices, mask):
    masked_slices = []
    for index in range(raw_slices.shape[0]):
        masked_slices.append(mask[index, :, :]*raw_slices[index, :, :])

    return np.array(masked_slices)


def image_preprocessing(raw_slices, display=False):
    enhanced_slices = contrast_enhancement(raw_slices)

    masks = __skull_strip_mcstrip_algorithm(enhanced_slices, display)

    no_skull_slices = apply_mask(raw_slices, masks["consensus"])
    return masks, no_skull_slices


def image_preprocessing_brainweb(raw_slices, display=False):
    masks = __skull_strip_mcstrip_algorithm(raw_slices, display)

    no_skull_slices = apply_mask(raw_slices, masks["consensus"])
    return masks, no_skull_slices


def mri_standarization(raw_slices, subjects, display=False):
    s1 = 0
    s2 = 4096
    pc1 = 0
    pc2 = 99.8

    landmark_file_path = os.path.join('utils', 'landmarks.txt')
    if not os.path.exists(landmark_file_path):
        x_array = []
        # Landmark extraction
        for s in subjects:
            m1 = np.min(raw_slices[s])
            m2 = np.max(raw_slices[s])
            [hist, edges] = np.histogram(raw_slices[s], bins=m2)
            p1 = np.percentile(raw_slices[s], pc1)
            p2 = np.percentile(raw_slices[s], pc2)
            peaks = find_peaks(hist, prominence=1, width=3,
                               height=700, distance=3)
            x = np.max(edges[peaks[0]])
            x_p = s1 + (((x - p1)/(p2 - p1)) * (s2 - s1))
            x_array.append(x_p)
            if display:
                img_utils.plot_hist_peaks(s, hist, peaks)

        # Store landmarks
        x_p_mean = np.mean(x_array)

        with open(landmark_file_path, 'w') as landmark_file:
            landmark_file.write(str(x_p_mean))
    else:
        with open(landmark_file_path, 'r') as landmark_file:
            x_p_mean = float(landmark_file.read())

    # Apply transformations
    for s in subjects:
        m1 = np.min(raw_slices[s])
        m2 = np.max(raw_slices[s])
        [hist, edges] = np.histogram(raw_slices[s], bins=m2)
        p1 = np.percentile(raw_slices[s], pc1)
        p2 = np.percentile(raw_slices[s], pc2)
        peaks = find_peaks(hist, prominence=1, width=3,
                           height=700, distance=3)
        x = np.max(edges[peaks[0]])
        for index in range(raw_slices[s].shape[0]):
            for row in range(raw_slices[s].shape[1]):
                for col in range(raw_slices[s].shape[2]):
                    voxel_val = raw_slices[s][index, row, col]

                    if voxel_val >= m1 and voxel_val <= x:
                        new_voxel_val = x_p_mean + \
                            (voxel_val - x)*((s1 - x_p_mean)/(p1 - x))
                    else:
                        new_voxel_val = x_p_mean + \
                            (voxel_val - x)*((s2 - x_p_mean)/(p2 - x))

                    raw_slices[s][index, row, col] = new_voxel_val

        if display:
            [hist, edges] = np.histogram(raw_slices[s], bins=m2)
            peaks = find_peaks(hist, prominence=1, width=3,
                               height=700, distance=3)
            img_utils.plot_hist_peaks(s, hist, peaks)
            img_utils.plot_stack(s, raw_slices[s])

    return raw_slices
