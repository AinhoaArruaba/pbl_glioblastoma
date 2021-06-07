import utils.img_utils as img_utils

import numpy as np
from sklearn.mixture import GaussianMixture
from skimage import morphology
from skimage import measure
from scipy import ndimage
from skimage import filters
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import math
import feature_extraction as fe


def get_tumor_segmentation(mri_slices_preprocessed, type, display):
    mri_slices_segmented = []
    mri_slices_masks = []
    mri_slices_masks_validated = []

    for index in enumerate(mri_slices_preprocessed):
        segmented_region, segmented_mask = segment_tumors(
            mri_slices_preprocessed[index[0], :, :], display)
        segmented_region = segmented_region.astype(np.int16)
        segmented_mask = segmented_mask.astype(np.int16)
        mri_slices_segmented.append(segmented_region)
        mri_slices_masks.append(segmented_mask)

    mri_slices_masks_validated = validate_masks(
        mri_slices_preprocessed, mri_slices_masks, type)
    mri_slices_segmented = [
        seg*val for seg, val in zip(mri_slices_segmented, mri_slices_masks_validated)]

    if display:
        img_utils.plot_stack('Detected tumors', np.array(mri_slices_segmented))

    return mri_slices_segmented, mri_slices_masks, mri_slices_masks_validated


def validate_masks(mri_slices_preprocessed, mri_slices_masks, type):
    validated_masks = mri_slices_masks[:]
    mri_masks_continuity = np.zeros(len(validated_masks))
    mri_masks_area = np.zeros(len(validated_masks))
    max_slice_area_idx = 0
    slice_area_max = 0
    # Check if the masks are valid
    for index in enumerate(validated_masks):
        all_labels = measure.label(validated_masks[index[0]])
        regions = measure.regionprops(all_labels)
        slice_area = fe.compute_slice_area(validated_masks[index[0]])

        regions.sort(key=lambda region: region.area, reverse=True)

        input_mask = mri_slices_preprocessed[index[0]] > 0
        input_labels = measure.label(input_mask)
        input_regions = measure.regionprops(input_labels)
        totalArea = sum(reg.area for reg in input_regions)

        slice_area = sum(reg.area for reg in regions)

        validated_masks[index[0]], slice_area = __check_mask_validity(
            regions, totalArea, all_labels, validated_masks[index[0]], slice_area, type)

        if slice_area > slice_area_max:
            slice_area_max = slice_area
            max_slice_area_idx = index[0]
        mri_masks_area[index[0]] = slice_area
        validated_masks[index[0]
                         ] = validated_masks[index[0]].astype(np.int16)

    # Compute continuity of the tumor among slices
    cont = 0
    for mri_slice_mask_cur, mri_slice_mask_next in zip(validated_masks, validated_masks[1:]):
        mri_masks_continuity[cont] = sum(
            sum(mri_slice_mask_cur*mri_slice_mask_next))
        cont = cont + 1

    # Select slices that contain the tumor
    validated_masks = __select_tumor_slices(mri_masks_continuity,
                                             validated_masks, max_slice_area_idx, mri_masks_area)

    return validated_masks


def __check_mask_validity(regions, totalArea, all_labels, mri_slice_mask, slice_area, type):
    if type == 1:
        compactness_limit = 0.65
        eccentricity_limit = 0.90
        totalArea_limit = 0.4
    else:
        compactness_limit = 0.75
        eccentricity_limit = 0.95
        totalArea_limit = 0.4
    
    for reg_idx in enumerate(regions):
        compactness = 1 - 4*math.pi * \
            regions[reg_idx[0]].area/(regions[reg_idx[0]].perimeter**2)
        if (compactness > compactness_limit or regions[reg_idx[0]].eccentricity > eccentricity_limit 
            or regions[reg_idx[0]].area > totalArea*totalArea_limit):
            mask = mri_slice_mask
            mask = mask - \
                np.where(all_labels == regions[reg_idx[0]].label, 1, 0)
            mri_slice_mask = mask
            slice_area = slice_area - regions[reg_idx[0]].area

    return mri_slice_mask, slice_area


def __select_tumor_slices(mri_masks_continuity, mri_slices_masks, max_slice_area_idx, mri_masks_area):
    if np.count_nonzero(mri_masks_continuity) == 0:
        for index in enumerate(mri_slices_masks):
            if index[0] != max_slice_area_idx:
                mri_slices_masks[index[0]][:] = 0
    else:
        first_idx, max_num_connected = __get_first_slice_idx(
            mri_masks_continuity, mri_masks_area)
        first_idx, max_num_connected = __check_previous_next_slides(mri_slices_masks, mri_masks_continuity,
                                                                    first_idx, max_num_connected)

        for index in enumerate(mri_slices_masks):
            if index[0] < first_idx or index[0] > (first_idx+max_num_connected):
                mri_slices_masks[index[0]][:] = 0

    return mri_slices_masks


def __get_first_slice_idx(mri_masks_continuity, mri_masks_area):
    first_idx = 0
    max_num_connected = 0
    max_connected_area = 0
    nz_idxs = np.nonzero(mri_masks_continuity)
    for index in nz_idxs[0]:
        num_connected = 1
        connected_area = mri_masks_area[index] + \
            mri_masks_area[index+num_connected]
        while mri_masks_continuity[index+num_connected]:
            num_connected = num_connected + 1
            connected_area = connected_area + \
                mri_masks_area[index+num_connected]
        if num_connected > max_num_connected:
            max_num_connected = num_connected
            first_idx = index
            max_connected_area = connected_area
        if num_connected == max_num_connected and connected_area > max_connected_area:
            max_num_connected = num_connected
            first_idx = index
            max_connected_area = connected_area

    return first_idx, max_num_connected


def __check_previous_next_slides(mri_slices_masks, mri_masks_continuity, first_idx, max_num_connected):
    previous_idx = first_idx - 2
    if previous_idx > 0 and sum(sum(mri_slices_masks[first_idx]*mri_slices_masks[previous_idx])) > 0:
        first_idx = previous_idx
        max_num_connected = max_num_connected + 2
        while first_idx-1 > 0 and mri_masks_continuity[first_idx-1] > 0:
            first_idx = first_idx - 1
            max_num_connected = max_num_connected + 1

    next_idx = first_idx + max_num_connected + 2
    if next_idx < len(mri_slices_masks) and sum(sum(mri_slices_masks[first_idx+max_num_connected]*mri_slices_masks[next_idx])) > 0:
        max_num_connected = max_num_connected + 2
        while next_idx+1 < len(mri_slices_masks) and mri_masks_continuity[next_idx+1] > 0:
            next_idx = next_idx - 1
            max_num_connected = max_num_connected + 1

    return first_idx, max_num_connected


def segment_tumors(mri_slice_preprocessed, display):
    mri_slice = mri_slice_preprocessed

    currentSlice = np.array(mri_slice)

    # Keep largest connected region
    currentSlice_region = __keepLargestRegion(currentSlice)

    # Mixture of Gaussians: black background, high intensive brain parts and rest of the brain
    maxMu = __mixture_of_gaussians(currentSlice_region)

    # Greyscale morphological reconstruction
    slice_subs = __morph_reconstruction(currentSlice_region, maxMu)

    # Binary adaptive thresholding
    binary_slice = __binary_thresholding(slice_subs)

    # Region with maximum mean intensity is marked as tumour
    tumor_mask = __create_tumor_mask(binary_slice, slice_subs)

    # Morphological operations
    tumor_mask_final = __morph_op(tumor_mask)

    segmented_tumor = tumor_mask_final*mri_slice

    if display:
        img_utils.display_seg_results(currentSlice, slice_subs,
                                  binary_slice, tumor_mask,
                                 tumor_mask_final, segmented_tumor)

    return segmented_tumor, tumor_mask_final


def __keepLargestRegion(currentSlice):
    binary_slice = currentSlice > 0
    binary_slice = morphology.erosion(binary_slice, morphology.square(5))
    labels = measure.label(binary_slice)
    regions = measure.regionprops(labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    region_mask = np.ndarray(
        [currentSlice.shape[0], currentSlice.shape[1]], dtype=np.int16)
    region_mask[::] = 0
    if len(regions) > 0:
        region_mask = region_mask + \
            np.where(labels == regions[0].label, 1, 0)
        region_mask = region_mask.astype(np.int16)
    currentSlice = currentSlice*region_mask

    return currentSlice


def __mixture_of_gaussians(currentSlice):
    noOfGaussians = 3

    currentSliceVector = np.array(currentSlice.flatten())
    sliceTransposed = np.array([currentSliceVector])
    sliceTransposed = sliceTransposed.T

    gm = GaussianMixture(n_components=noOfGaussians,
                         random_state=0, n_init=3).fit(sliceTransposed)
    gm_means = np.array(gm.means_).flatten()
    gm_covariances = np.array(gm.covariances_).flatten()

    maxV = np.max(gm_covariances)
    maxInd = np.where(gm_covariances == maxV)
    maxMu = max(gm_means[maxInd])

    return maxMu


def __morph_reconstruction(currentSlice, maxMu):
    seedImage = currentSlice - maxMu
    slice_rec = morphology.reconstruction(
        seedImage, currentSlice, method='dilation')
    slice_subs = currentSlice - slice_rec  # get local maximas

    return slice_subs


def __binary_thresholding(slice_subs):
    thres = filters.threshold_yen(slice_subs)
    binary_slice = slice_subs > thres

    return binary_slice


def __create_tumor_mask(binary_slice, slice_subs):
    binary_slice_morph = __morph_op_initial_mask(binary_slice)

    labels = measure.label(binary_slice_morph)
    regions = measure.regionprops(labels)

    if len(regions) != 0:
        regions_idx = __find_tumor_region(slice_subs, labels, regions)

    tumor_mask = np.ndarray(
        [binary_slice.shape[0], binary_slice.shape[1]], dtype=np.int16)
    tumor_mask[:] = 0
    if regions_idx.size > 0:
        # +1 because the labels start from 1 and the indexes from 0
        regions_idx_list = regions_idx.tolist()
        if regions_idx.size == 1:
            regions_idx_list = [regions_idx_list]
        for idx in regions_idx_list:
            tumor_mask = tumor_mask + np.where(labels == idx+1, 1, 0)

    return tumor_mask


def __find_tumor_region(slice_subs, labels, regions):
    regions_meanInt, regions_meanInt_possible = __compute_mean_intensities(
        slice_subs, labels, regions)
    region_idx = __select_regions(
        regions, regions_meanInt, regions_meanInt_possible)
    return region_idx


def __compute_mean_intensities(slice_subs, labels, regions):
    regions_meanInt = np.zeros(len(regions))
    regions_meanInt_possible = np.zeros(len(regions))
    for region_idx in range(len(regions)):
        locs = np.where(labels == (region_idx+1))
        pixels = slice_subs[locs]
        region_mean = np.mean(pixels)
        regions_meanInt_possible[region_idx] = region_mean
        if regions[region_idx].area > 150:  # set minimum area
            regions_meanInt[region_idx] = region_mean

    return regions_meanInt, regions_meanInt_possible


def __select_regions(regions, regions_meanInt, regions_meanInt_possible):
    maxInt = np.max(regions_meanInt)
    maxIdx = max(max(np.where(regions_meanInt == maxInt)))
    max_centroid = regions[maxIdx].centroid
    regions_meanInt[maxIdx] = 0
    regions_meanInt_possible[maxIdx] = 0

    intThreshold = np.floor(maxInt*0.8)
    possibleRegions = max(np.where(regions_meanInt_possible >= intThreshold))

    selected_regions = np.array(maxIdx)

    for possibleIdx in range(len(possibleRegions)):
        possible_centroid = regions[possibleRegions[possibleIdx]].centroid
        euc_dist = np.sqrt((max_centroid[0] - possible_centroid[0])**2 +
                           (max_centroid[1] - possible_centroid[1])**2)
        if euc_dist < 75:
            selected_regions = np.append(
                selected_regions, possibleRegions[possibleIdx])

    return selected_regions


def __morph_op(tumor_mask):
    tumor_mask = morphology.dilation(tumor_mask, morphology.square(5))
    tumor_mask = morphology.closing(tumor_mask, morphology.square(6))
    tumor_mask = ndimage.binary_fill_holes(tumor_mask)
    tumor_mask = morphology.erosion(tumor_mask, morphology.square(1))

    return tumor_mask


def __morph_op_initial_mask(tumor_mask):
    tumor_mask = morphology.dilation(tumor_mask, morphology.square(3))
    tumor_mask = morphology.closing(tumor_mask, morphology.square(3))
    tumor_mask = morphology.erosion(tumor_mask, morphology.square(1))

    return tumor_mask


def get_edema_segmentation(mri_slices_preprocessed, m_t_A_t2, display):
    mri_slices_segmented = []
    mri_slices_masks = []

    if len(mri_slices_preprocessed) != len(m_t_A_t2):
        m_t_A_t2 = []
        for index in enumerate(mri_slices_preprocessed):
            segmented_region, segmented_mask = segment_edema_flair(
                mri_slices_preprocessed[index[0], :, :], m_t_A_t2, display)
            mri_slices_segmented.append(segmented_region)
            mri_slices_masks.append(segmented_mask)
    else:
        for index in enumerate(mri_slices_preprocessed):
            segmented_region, segmented_mask = segment_edema_flair(mri_slices_preprocessed[index[0], :, :],
                                                                   m_t_A_t2[index[0], :, :], display)
            mri_slices_segmented.append(segmented_region)
            mri_slices_masks.append(segmented_mask)

    if display:
        img_utils.plot_stack('Detected edemas', np.array(mri_slices_segmented))

    return mri_slices_segmented, mri_slices_masks


def segment_edema_flair(mri_slice_preprocessed, m_t_A_t2, display):
    mri_slice = mri_slice_preprocessed

    currentSlice = np.array(mri_slice)

    # Keep largest connected region
    currentSlice_region = __keepLargestRegion(currentSlice)

    # Compute thresholds
    threshold_A, threshold_B, threshold_fgr = __compute_threshold_flair(
        currentSlice_region, display)

    # Obtain masks
    m_t_A_flair = currentSlice_region > threshold_A
    m_t_B_flair = currentSlice_region > threshold_B
    m_t_fgr_flair = currentSlice_region > threshold_fgr

    # Segmentation
    edema_mask = morphology.reconstruction(
        m_t_B_flair, m_t_A_flair, method='dilation', selem=None, offset=None)

    if len(m_t_A_t2) > 0:
        m_cleaned = m_t_A_t2*m_t_fgr_flair
        if np.max(m_cleaned) > 0:
            SE = np.ones((3, 3))
            marker_result = edema_mask
            marker = edema_mask
            marker[:] = 0
            mask = m_cleaned
            while marker.all() != marker_result.all():
                marker = marker_result
                marker_dilated = morphology.dilation(
                    marker, morphology.square(3))
                marker_result = marker_dilated*mask
            edema_mask = marker_result

    segmented_edema = edema_mask*mri_slice

    # if display:
    #   img_utils.display_seg_results(currentSlice, m_t_A_flair,
    #                              m_t_B_flair, m_t_fgr_flair,
    #                             edema_mask, segmented_edema)

    return segmented_edema, edema_mask


def __compute_threshold_flair(currentSlice_region, display):
    m2 = np.max(currentSlice_region)

    threshold_A = 0
    threshold_B = 0
    threshold_fgr = 0

    # Check it is not empty
    if m2 > 0:
        [hist, edges] = np.histogram(currentSlice_region, bins=m2)

        # Filter
        hist_filtered = savgol_filter(hist, 101, 2)

        # Get peaks
        peaks = find_peaks(hist_filtered, prominence=1, width=3, distance=3)

        mode1_idx, mode2_idx = __get_modes(hist_filtered, peaks)

        # Compute thresholds
        threshold_A, threshold_B, threshold_fgr = __compute_thresholds(
            mode1_idx, mode2_idx, hist_filtered, edges)
        histPeaks = np.array(
            [mode1_idx, mode2_idx, threshold_A, threshold_B, threshold_fgr])
        if display:
            img_utils.plot_hist_peaks(
                'Prueba', hist_filtered, histPeaks.astype(np.int64))

    return threshold_A, threshold_B, threshold_fgr


def __get_modes(hist_filtered, peaks):
    mode1 = 0
    mode2 = 0

    peak_array = peaks[0]
    if peak_array.size > 0:
        peak_idx = peak_array[0]
        mode1 = hist_filtered[peak_idx]
    if peak_array.size > 1:
        peak_idx = peak_array[1]
        mode2 = hist_filtered[peak_idx]

    # Check if the mode is on the first values
    first_idxs_max = np.max(hist_filtered[0:3])
    if first_idxs_max > mode2:
        mode2 = first_idxs_max
        if mode2 > mode1:
            mode2 = mode1
            mode1 = first_idxs_max
    mode1_idx = min(np.where(hist_filtered == mode1))[0]
    mode2_idx = min(np.where(hist_filtered == mode2))[0]

    return mode1_idx, mode2_idx


def __compute_thresholds(mode1_idx, mode2_idx, hist_filtered, edges):
    max_slope = 0
    max_slope_idx = mode2_idx
    cur_slope = 0
    prev_slope = 1
    cur_slope_idx = mode2_idx
    while prev_slope > 0 and cur_slope_idx < len(hist_filtered)-1:
        cur_slope = hist_filtered[cur_slope_idx] - \
            hist_filtered[cur_slope_idx+1]
        if cur_slope > max_slope:
            max_slope = cur_slope
            max_slope_idx = cur_slope_idx
        cur_slope_idx = cur_slope_idx + 1
        prev_slope = cur_slope
        # Check slope continuity
        if prev_slope <= 0:
            slope_range = hist_filtered[cur_slope_idx-1: cur_slope_idx+4]
            if min(slope_range) != hist_filtered[cur_slope_idx-1]:
                prev_slope = 1

    # Threshold A
    threshold_A = edges[max_slope_idx]
    # Threshold B
    threshold_B = edges[cur_slope_idx-1]

    cur_slope = 0
    prev_slope = 1
    cur_slope_idx = mode2_idx
    while prev_slope > 0 and cur_slope_idx > 0:
        cur_slope = hist_filtered[cur_slope_idx] - \
            hist_filtered[cur_slope_idx-1]
        cur_slope_idx = cur_slope_idx - 1
        prev_slope = cur_slope
        if cur_slope_idx == 52:
            aa = 1
        # Check slope continuity
        if prev_slope <= 0:
            slope_range = hist_filtered[cur_slope_idx-3: cur_slope_idx+2]
            if min(slope_range) != hist_filtered[cur_slope_idx+1]:
                prev_slope = 1

    # Threshold Fgr
    threshold_fgr = edges[cur_slope_idx+1]

    return threshold_A, threshold_B, threshold_fgr


def get_t2_mask(mri_slices_preprocessed, display):
    ref_slice = mri_slices_preprocessed[0]
    pixel_dims = (len(mri_slices_preprocessed),  int(
        ref_slice.shape[0]), int(ref_slice.shape[1]))
    t2_masks = np.zeros(pixel_dims, dtype=ref_slice.dtype)

    for index in enumerate(mri_slices_preprocessed):
        t2_masks[index[0], :, :] = __get_thres_A_mask(
            mri_slices_preprocessed[index[0], :, :], display)

    if display:
        img_utils.plot_stack('T2 masks', np.array(t2_masks))

    return t2_masks


def __get_thres_A_mask(mri_slice_preprocessed, display):
    mri_slice = mri_slice_preprocessed

    currentSlice = np.array(mri_slice)

    # Keep largest connected region
    currentSlice_region = __keepLargestRegion(currentSlice)

    # Compute thresholds
    threshold_A, threshold_B, threshold_fgr = __compute_threshold_flair(
        currentSlice_region, display)

    # Obtain mask
    m_t_A = currentSlice_region > threshold_A

    return m_t_A


def save_tumor_contours(fig_name, s, mri_slices_preprocessed, mri_slices_masks):
    img_utils.save_stack_contours(
        fig_name, s, mri_slices_preprocessed, mri_slices_masks)
