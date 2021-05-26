
import numpy as np

from skimage import measure
from scipy import ndimage
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import entropy
from skimage.feature.texture import greycomatrix, greycoprops
import math

def extract_features(mri_slices_segmented, mri_slices_masks, mri_slices_std, slice_thicknesses):
    # Find first and last index of valid masks
    first_idx, last_idx = __get_first_last_idx(mri_slices_masks)

    # Get tumor volume
    tumor_volume = 0
    tumor_volume, tumor_volume_slices = __get_tumor_volume(mri_slices_masks, slice_thicknesses, first_idx, last_idx)

    # Extract features from the mask with the highest tumor volume
    maxV = np.max(tumor_volume_slices)
    max_idx = np.where(tumor_volume_slices == maxV)[0][0] + first_idx
    dists = __extract_tumor_location(mri_slices_masks[max_idx], mri_slices_std[max_idx])
    slice_features = __extract_slice_features(mri_slices_segmented[max_idx], mri_slices_masks[max_idx])

    volume_dist_features = np.append(tumor_volume, dists)
    extracted_features = np.append(volume_dist_features, slice_features)

    return extracted_features

def __get_first_last_idx(mri_slices_masks):
    first_idx = -1
    last_idx = -1
    s_idx = 0
    e_idx = len(mri_slices_masks) - 1
    while first_idx == -1 or last_idx == -1:
        if mri_slices_masks[s_idx].any():
            first_idx = s_idx
        else:
            s_idx = s_idx + 1
        if mri_slices_masks[e_idx].any():
            last_idx = e_idx
        else:
            e_idx = e_idx - 1
        if s_idx > len(mri_slices_masks)-1 or e_idx < 0:
            first_idx = 0
            last_idx = 0

    return first_idx, last_idx

def __get_tumor_volume(mri_slices_masks, slice_thicknesses, first_idx, last_idx):
    tumor_volume_slices = np.zeros(last_idx-first_idx+1)
    for index in range(first_idx, last_idx+1):
        tumor_volume_slices[index-first_idx] = __compute_mask_volume(mri_slices_masks[index], slice_thicknesses[index])
        if tumor_volume_slices[index-first_idx] == 0 and first_idx != 0:
            tumor_volume_slices[index-first_idx] = __estimate_mask_volume(mri_slices_masks[index-1], 
                mri_slices_masks[index+1], slice_thicknesses[index])
    
    tumor_volume = sum(tumor_volume_slices)

    return tumor_volume, tumor_volume_slices

def __compute_mask_volume(mri_slice_mask, slice_thickness):
    slice_area = compute_slice_area(mri_slice_mask)
    slice_volume = slice_area*slice_thickness

    return slice_volume

def __estimate_mask_volume(mri_previous_slice_mask, mri_next_slice_mask, slice_thickness):
    previous_slice_area = compute_slice_area(mri_previous_slice_mask)
    next_slice_area = compute_slice_area(mri_next_slice_mask)
    estimated_area = (previous_slice_area+next_slice_area)/2
    estimated_volume = estimated_area*slice_thickness

    return estimated_volume

def compute_slice_area(mri_slice_mask):
    slice_area = 0
    all_labels = measure.label(mri_slice_mask)
    regions = measure.regionprops(all_labels)
    if len(regions) > 0:
        slice_area = sum(reg.area for reg in regions)

    return slice_area

def __extract_tumor_location(mri_slice_mask, mri_slice_std):
    labels = measure.label(mri_slice_mask)
    regions = measure.regionprops(labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    dist_x = float("nan")
    dist_y = float("nan")
    if len(regions) > 0:
        seg_centroid = regions[0].centroid

        mri_slice_std_bin = mri_slice_std > 0
        labels_std = measure.label(mri_slice_std_bin)
        regions_std = measure.regionprops(labels_std)
        regions_std.sort(key=lambda region: region.area, reverse=True)
        std_centroid = regions_std[0].centroid

        dist_x = seg_centroid[1] - std_centroid[1]
        dist_y = seg_centroid[0] - std_centroid[0]
    dists = [dist_x, dist_y]

    return dists

def __extract_slice_features(mri_slice_segmented, mri_slice_mask):
    extracted_features = []

    # Mean compactness
    compactness = __compute_compactness(mri_slice_mask)
    extracted_features.append(compactness)

    # Compute mean grey level
    uf, px_smooth = __compute_mean_grey_level(mri_slice_segmented)
    extracted_features.append(uf)

    # Standard deviation
    std = ndimage.standard_deviation(mri_slice_segmented)
    extracted_features.append(std)

    # Variance
    var = ndimage.variance(mri_slice_segmented)
    extracted_features.append(var)

    # Skewness
    skewness = skew(mri_slice_segmented, axis=None)
    extracted_features.append(skewness)

    # Kurtosis
    kur = kurtosis(mri_slice_segmented, fisher=True, axis=None)
    extracted_features.append(kur)

    # Entropy
    ent = entropy(mri_slice_segmented, base=2, axis=None)
    extracted_features.append(ent)

    # Smoothness
    smoothness = 1 - (1 / (1 + var))
    extracted_features.append(smoothness)

    # Uniformity
    uniformity = np.sum(px_smooth**2)
    extracted_features.append(uniformity)

    # GLCM properties
    glcm_properties = __compute_GLCM_properties(mri_slice_segmented)
    extracted_features = extracted_features + glcm_properties

    # Compute Hu moments
    M_hu = __compute_Hu_moments(mri_slice_segmented)
    extracted_features = np.append(extracted_features, M_hu)

    return extracted_features 

def __compute_compactness(mri_slice_mask):
    all_labels = measure.label(mri_slice_mask)
    regions = measure.regionprops(all_labels)
    compactness = float("nan")
    if len(regions) > 0:
        compactness = 0
        for reg_idx in enumerate(regions):
            compactness = compactness + 1 - 4*math.pi*regions[reg_idx[0]].area/(regions[reg_idx[0]].perimeter**2)
        compactness = compactness / len(regions)

    return compactness

def __compute_mean_grey_level(mri_slice_segmented):
    uf = 0
    px_smooth = 0
    m2 = np.max(mri_slice_segmented)
    if m2 > 0:
        [hist, edges] = np.histogram(mri_slice_segmented, bins=m2+1)

        px_smooth = hist / np.sum(hist)
        uf = sum(hist*edges[1:len(edges)])

    return uf, px_smooth

def __compute_GLCM_properties(mri_slice_segmented):
    glcm_properties = []

    # It is necessary to convert the image to uint8 to use the function
    mri_slice_segmented_uint8 = mri_slice_segmented*255/np.max(mri_slice_segmented)
    mri_slice_segmented_uint8 = mri_slice_segmented_uint8.astype(np.uint8)

    # GLCM considering a horizontal offset of 5
    glcm = greycomatrix(mri_slice_segmented_uint8, distances=[5], angles=[0], 
        levels=256, symmetric=True, normed=True)

    # Properties
    glcm_properties.append(greycoprops(glcm, 'contrast')[0][0])
    glcm_properties.append(greycoprops(glcm, 'dissimilarity')[0][0])
    glcm_properties.append(greycoprops(glcm, 'homogeneity')[0][0])
    glcm_properties.append(greycoprops(glcm, 'energy')[0][0])
    glcm_properties.append(greycoprops(glcm, 'correlation')[0][0])
    glcm_properties.append(greycoprops(glcm, 'ASM')[0][0])

    return glcm_properties

def __compute_Hu_moments(mri_slice_segmented):
    M_c = measure.moments_central(mri_slice_segmented)
    nu = measure.moments_normalized(M_c)
    M_hu = measure.moments_hu(nu)

    return M_hu
