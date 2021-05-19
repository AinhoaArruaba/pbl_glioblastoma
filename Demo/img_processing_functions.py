import utils.img_utils as img_utils

import numpy as np
import os
import json

from skimage import morphology
from skimage import measure
from skimage import filters
from skimage import exposure
from skimage import feature

from sklearn import cluster

from medpy.filter import smoothing

from scipy import ndimage
from scipy.signal import find_peaks


# Skull stripping algorithm implementation for 3D brain MRI based in McStrip algorithm
def __check_slice_connectivity(mask_slices):
    pass


def __get_brain_coarse_mask(raw_slices):
    coarse_masks = []
    # Thresholding with Otsu's method
    threshold = filters.threshold_otsu(raw_slices)
    # Apply threshold to obtain binary slice
    coarse_masks = raw_slices > threshold

    coarse_masks = morphology.closing(
        np.array(coarse_masks), morphology.ball(2))

    coarse_masks = ndimage.binary_fill_holes(np.array(coarse_masks))
    for index in enumerate(coarse_masks):
        coarse_masks[index[0], :, :] = ndimage.binary_fill_holes(
            coarse_masks[index[0], :, :])

    return np.array(coarse_masks)


def __get_brain_kmeans_mask(raw_slices, coarse_masks):
    # Thresholding with Otsu's method
    t = filters.threshold_otsu(raw_slices)
    coarse_masks = morphology.erosion(coarse_masks, morphology.ball(2))
    # Elliminate background
    slices_no_background = apply_mask(raw_slices, coarse_masks)

    # Create clusters
    X = slices_no_background.reshape((-1, 1))
    k_m = cluster.KMeans(n_clusters=3)
    k_m.fit(X)
    values = k_m.cluster_centers_.squeeze()
    labels = k_m.labels_

    # Create the segmented array from labels and values
    slices_no_background_clusters = np.choose(labels, values)
    # Reshape the array as the original image
    slices_no_background_clusters.shape = slices_no_background.shape

    # Select the brain tissue mask
    slices_brain_tissue = np.logical_and(np.logical_and(
        slices_no_background_clusters < t*2, slices_no_background_clusters > min(values)), coarse_masks)
    slices_non_brain_tissue = np.logical_or(
        slices_no_background_clusters > t*2, slices_no_background_clusters < min(values))
    slices_brain_tissue = morphology.erosion(
        slices_brain_tissue, morphology.ball(1))

    for index in enumerate(slices_non_brain_tissue):
        slc = slices_non_brain_tissue[index[0], :, :]
        slc = morphology.opening(slc, morphology.diamond(2))
        all_labels = measure.label(slc)
        regions = measure.regionprops(all_labels)
        min_compactness = 55
        tumor_label = []
        for reg in regions:
            compactness = reg.perimeter ** 2 / reg.area
            if compactness < min_compactness and reg.area > 500:
                tumor_label.append(reg.label)
                # min_compactness = compactness
        if tumor_label:
            for l in tumor_label:
                slc = np.logical_and(slc, np.logical_not(
                    np.where(all_labels == l, 1, 0)))

        slc = morphology.closing(slc, morphology.diamond(2))
        slices_non_brain_tissue[index[0], :, :] = slc

    a_mask = np.logical_and(np.logical_and(np.logical_and(np.logical_not(slices_non_brain_tissue), coarse_masks), np.logical_and(
        np.logical_not(slices_brain_tissue), coarse_masks)), np.logical_not(slices_no_background_clusters == min(values)))
    a_mask = morphology.opening(a_mask, morphology.ball(1))
    kmeans_mask = np.logical_and(np.logical_and(np.logical_not(
        slices_non_brain_tissue), slices_brain_tissue), coarse_masks)
    kmeans_mask = np.logical_and(
        np.logical_or(kmeans_mask, a_mask), coarse_masks)

    for index in enumerate(kmeans_mask):
        slc = kmeans_mask[index[0], :, :]
        slc = morphology.erosion(slc, morphology.disk(1))
        slc = morphology.closing(slc, morphology.disk(4))
        slc = morphology.opening(slc, morphology.disk(2))
        slc = ndimage.binary_fill_holes(slc)
        slc = morphology.closing(slc, morphology.disk(8))
        slc = ndimage.binary_fill_holes(slc)
        slc = ndimage.binary_fill_holes(slc)
        kmeans_mask[index[0], :, :] = slc

    kmeans_mask = morphology.opening(kmeans_mask, morphology.ball(1))

    all_labels = measure.label(kmeans_mask)
    regions = measure.regionprops(all_labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    kmeans_mask = np.ndarray(
        [raw_slices.shape[0], raw_slices.shape[1], raw_slices.shape[2]], dtype=np.int8)
    kmeans_mask[::] = 0
    if regions:
        kmeans_mask = kmeans_mask + \
            np.where(all_labels == regions[0].label, 1, 0)

    return kmeans_mask


def __get_brain_thresh_mask(raw_slices, coarse_masks):
    thresh_masks = []
    # Thresholding with Multi-Otsu's method
    lv1_filtered_slices = apply_mask(raw_slices, coarse_masks)

    black_tophat_slices = morphology.black_tophat(
        lv1_filtered_slices, morphology.ball(1))
    white_tophat_slices = morphology.white_tophat(
        lv1_filtered_slices, morphology.ball(1))

    for index in enumerate(lv1_filtered_slices):
        lv1_filtered_slices[index[0], :, :] = lv1_filtered_slices[index[0], :, :] - \
            black_tophat_slices[index[0], :, :] + \
            white_tophat_slices[index[0], :, :]

    lv1_filtered_slices = apply_mask(raw_slices, coarse_masks)

    thresholds = filters.threshold_multiotsu(lv1_filtered_slices, classes=3)
    # Apply threshold to obtain binary slice
    thresh_masks = np.logical_and(
        lv1_filtered_slices > thresholds[0], lv1_filtered_slices < thresholds[1])

    thresh_masks = ndimage.binary_fill_holes(thresh_masks)
    thresh_masks = np.array(thresh_masks)
    thresh_masks = morphology.opening(thresh_masks, morphology.octahedron(2))
    thresh_masks = morphology.erosion(thresh_masks, morphology.octahedron(1))

    thresh_masks = np.logical_and(
        thresh_masks, morphology.erosion(coarse_masks, morphology.ball(5)))

    for index in enumerate(thresh_masks):
        thresh_masks[index[0], :, :] = morphology.opening(
            thresh_masks[index[0], :, :], morphology.disk(1))
        thresh_masks[index[0], :, :] = morphology.closing(
            thresh_masks[index[0], :, :], morphology.disk(2))
        thresh_masks[index[0], :, :] = ndimage.binary_fill_holes(
            thresh_masks[index[0], :, :])

    all_labels = measure.label(thresh_masks)
    regions = measure.regionprops(all_labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    thresh_masks = np.ndarray(
        [raw_slices.shape[0], raw_slices.shape[1], raw_slices.shape[2]], dtype=np.int8)
    thresh_masks[::] = 0
    if regions:
        thresh_masks = thresh_masks + \
            np.where(all_labels == regions[0].label, 1, 0)

    return thresh_masks


def __get_brain_bse_mask(raw_slices, coarse_masks):
    # http://brainsuite.org/processing/surfaceextraction/bse/
    # Apply anisotropic difussion filter
    thresh_masks = []
    lv1_filtered_slices = apply_mask(raw_slices, coarse_masks)

    black_tophat_slices = morphology.black_tophat(
        lv1_filtered_slices, morphology.ball(1))
    white_tophat_slices = morphology.white_tophat(
        lv1_filtered_slices, morphology.ball(1))

    for index in enumerate(lv1_filtered_slices):
        lv1_filtered_slices[index[0], :, :] = lv1_filtered_slices[index[0], :, :] - 2*(
            black_tophat_slices[index[0], :, :]) + white_tophat_slices[index[0], :, :]

    lv1_filtered_slices = apply_mask(raw_slices, coarse_masks)

    thresholds = filters.threshold_multiotsu(lv1_filtered_slices, classes=3)
    # Apply threshold to obtain binary slice
    thresh_masks = np.logical_and(
        lv1_filtered_slices > thresholds[0], lv1_filtered_slices < thresholds[1])

    thresh_masks = ndimage.binary_fill_holes(thresh_masks)
    lv1_filtered_slices = apply_mask(lv1_filtered_slices, thresh_masks)

    slices_filtered_anisotropic = np.array(smoothing.anisotropic_diffusion(
        lv1_filtered_slices, niter=3))

    # Apply Marr-Hildreth filtering algorithm (laplacian of gaussian filter)
    slices_gauss_filt = filters.gaussian(slices_filtered_anisotropic, sigma=2)
    slices_laplacian_filt = filters.laplace(slices_gauss_filt, ksize=5)

    slices_log = ndimage.gaussian_laplace(slices_filtered_anisotropic, 2)
    slices_binary = slices_log > 2

    # slices_binary = np.logical_and(slices_filtered_log < 0, thresh_masks)
    slices_binary = morphology.closing(
        slices_binary, morphology.octahedron(1))

    slices_binary = morphology.dilation(
        slices_binary, morphology.octahedron(1))

    slices_binary = np.logical_and(np.logical_not(slices_binary), morphology.erosion(
        coarse_masks, morphology.octahedron(2)))

    slices_binary = ndimage.binary_fill_holes(slices_binary)

    for index in enumerate(slices_binary):
        slc = slices_binary[index[0], :, :]
        slc = morphology.opening(slc, morphology.disk(2))
        slc = morphology.closing(slc, morphology.disk(1))
        slc = ndimage.binary_fill_holes(slc)
        slices_binary[index[0], :, :] = slc

    bse_masks = morphology.opening(
        slices_binary, morphology.octahedron(1))

    bse_masks = morphology.closing(
        slices_binary, morphology.octahedron(2))

    bse_masks = ndimage.binary_fill_holes(bse_masks)

    bse_masks = np.logical_and(bse_masks, morphology.erosion(
        coarse_masks, morphology.octahedron(5)))

    all_labels = measure.label(bse_masks)
    regions = measure.regionprops(all_labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    bse_masks = np.ndarray(
        [raw_slices.shape[0], raw_slices.shape[1], raw_slices.shape[2]], dtype=np.int8)
    bse_masks[::] = 0
    if regions:
        bse_masks = bse_masks + np.where(all_labels == regions[0].label, 1, 0)

    return bse_masks


def __get_consensus_mask(coarse_masks, thresh_masks, kmeans_mask, bse_masks):
    consensus_mask = np.empty(coarse_masks.shape, dtype=bool)
    for index in range(consensus_mask.shape[0]):
        for row in range(consensus_mask.shape[1]):
            for col in range(consensus_mask.shape[2]):
                vals = [thresh_masks[index, row, col],
                        kmeans_mask[index, row, col]*2, bse_masks[index, row, col]]
                if (coarse_masks[index, row, col] and sum(vals) > 2):
                    consensus_mask[index, row, col] = True
                else:
                    consensus_mask[index, row, col] = False

    # Fill holes mask
    consensus_mask = morphology.opening(
        consensus_mask, morphology.octahedron(1))
    all_labels = measure.label(consensus_mask)
    regions = measure.regionprops(all_labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    consensus_mask = np.ndarray(
        [consensus_mask.shape[0], consensus_mask.shape[1], consensus_mask.shape[2]], dtype=np.int8)
    consensus_mask[::] = 0
    if regions:
        consensus_mask = consensus_mask + \
            np.where(all_labels == regions[0].label, 1, 0)

    consensus_mask = morphology.closing(
        consensus_mask, morphology.octahedron(3))

    for index in range(consensus_mask.shape[0]):
        consensus_mask[index, :, :] = ndimage.binary_fill_holes(
            consensus_mask[index, :, :])

    return consensus_mask


def __skull_strip_mcstrip_algorithm(raw_slices, display=False):
    coarse_masks = __get_brain_coarse_mask(raw_slices)
    kmeans_masks = __get_brain_kmeans_mask(raw_slices, coarse_masks)
    thresh_masks = __get_brain_thresh_mask(raw_slices, coarse_masks)
    bse_masks = __get_brain_bse_mask(raw_slices, coarse_masks)
    consensus_mask = __get_consensus_mask(
        coarse_masks, thresh_masks, kmeans_masks, bse_masks)

    mask_dict = {"consensus": consensus_mask, "coarse": coarse_masks,
                 "threshold": thresh_masks, "kmeans": kmeans_masks, "bse": bse_masks}

    if display:
        img_utils.plot_stack('Coarse mask', coarse_masks)
        img_utils.plot_stack('Thresh mask', thresh_masks)
        img_utils.plot_stack('KMeans mask', kmeans_masks)
        img_utils.plot_stack('BSE mask', bse_masks)
        img_utils.plot_stack('Consensus mask', consensus_mask)

    return mask_dict


def apply_mask(raw_slices, mask):
    masked_slices = []
    for index in range(raw_slices.shape[0]):
        masked_slices.append(mask[index, :, :]*raw_slices[index, :, :])

    return np.array(masked_slices)


def image_skull_strip(raw_slices, display=False):
    masks = __skull_strip_mcstrip_algorithm(raw_slices, display)
    masks["consensus"] = masks["consensus"].astype(np.int16)
    no_skull_slices = apply_mask(raw_slices, masks["consensus"])
    no_skull_slices = no_skull_slices.astype(np.int16)

    return masks, no_skull_slices


def image_preprocessing_brainweb(raw_slices, display=False):
    masks = __skull_strip_mcstrip_algorithm(raw_slices, display)

    no_skull_slices = apply_mask(raw_slices, masks["consensus"])
    return masks, no_skull_slices


def standarization_landmark_extraction(raw_slices, subjects, s1, s2, pc1, pc2, landmark_file_path, display=False):
    x_array = {}
    # Landmark extraction
    for s in subjects:
        for mri_type in raw_slices[s]:
            m1 = np.min(raw_slices[s][mri_type])
            m2 = np.max(raw_slices[s][mri_type])
            [hist, edges] = np.histogram(raw_slices[s][mri_type], bins=m2)
            p1 = np.percentile(raw_slices[s][mri_type], pc1)
            p2 = np.percentile(raw_slices[s][mri_type], pc2)
            peaks = find_peaks(hist, prominence=1, width=3,
                               height=700, distance=3)
            x = np.max(edges[peaks[0]])
            x_p = s1 + (((x - p1)/(p2 - p1)) * (s2 - s1))
            if not mri_type in x_array:
                x_array[mri_type] = []
            x_array[mri_type].append(x_p)
            if display:
                img_utils.plot_hist_peaks(s, hist, peaks)

    # Store landmarks
    x_p_mean = {}
    for mri_type in x_array:
        x_p_mean[mri_type] = np.mean(x_array[mri_type])

    with open(landmark_file_path, 'w') as landmark_file:
        landmark_file.write(json.dumps(x_p_mean))

    return x_p_mean


def mri_standarization(raw_slices, subjects, landmark_filename, display=False):
    s1 = 0
    s2 = 4096
    pc1 = 0
    pc2 = 99.8

    landmark_file_path = os.path.join('utils', landmark_filename)
    if not os.path.exists(landmark_file_path):
        x_p_mean = standarization_landmark_extraction(
            raw_slices, subjects, s1, s2, pc1, pc2, landmark_file_path, display)
    else:
        with open(landmark_file_path, 'r') as landmark_file:
            x_p_mean = json.loads(landmark_file.read())

    # Apply transformations
    for s in subjects:
        for mri_type in raw_slices[s]:
            m1 = np.min(raw_slices[s][mri_type])
            m2 = np.max(raw_slices[s][mri_type])
            [hist, edges] = np.histogram(raw_slices[s][mri_type], bins=m2)
            p1 = np.percentile(raw_slices[s][mri_type], pc1)
            p2 = np.percentile(raw_slices[s][mri_type], pc2)
            peaks = find_peaks(hist, prominence=1, width=3,
                               height=700, distance=3)
            x = np.max(edges[peaks[0]])
            for index in range(raw_slices[s][mri_type].shape[0]):
                for row in range(raw_slices[s][mri_type].shape[1]):
                    for col in range(raw_slices[s][mri_type].shape[2]):
                        voxel_val = raw_slices[s][mri_type][index, row, col]

                        if voxel_val >= m1 and voxel_val <= x:
                            new_voxel_val = x_p_mean[mri_type] + (voxel_val - x) * (
                                (s1 - x_p_mean[mri_type])/(p1 - x))
                        else:
                            new_voxel_val = x_p_mean[mri_type] + (voxel_val - x) * (
                                (s2 - x_p_mean[mri_type])/(p2 - x))

                        raw_slices[s][mri_type][index,
                                                row, col] = new_voxel_val

            if display:
                [hist, edges] = np.histogram(raw_slices[s][mri_type], bins=m2)
                peaks = find_peaks(hist, prominence=1, width=3,
                                   height=700, distance=3)
                img_utils.plot_hist_peaks(s, hist, peaks)
                img_utils.plot_stack(s, raw_slices[s][mri_type])

    return raw_slices
