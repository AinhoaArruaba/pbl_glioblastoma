import utils.img_utils as img_utils

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage import morphology
from skimage import measure
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature
from skimage.filters import try_all_threshold

def segmentation(mri_slices_preprocessed, display):
    mri_slices_segmented = []
    mri_slices = mri_slices_preprocessed[1]

    for index in range(len(mri_slices)):
        currentSlice = np.array(mri_slices[index, :, :])

        # Mixture of Gaussians: black background, high intensive brain parts and rest of the brain
        maxMu = __mixture_of_gaussians(currentSlice)

        # Greyscale morphological reconstruction
        slice_subs = __morph_reconstruction(currentSlice, maxMu)

        # Binary adaptive thresholding
        binary_slice = __binary_thresholding(currentSlice, slice_subs, maxMu)        

        # Region with maximum mean intensity is marked as tumour
        tumor_mask = __create_tumor_mask(binary_slice, slice_subs)

        # Morphological operations
        tumor_mask_final = __final_morph_op(tumor_mask)

        segmented_tumor = tumor_mask_final*mri_slices[index, :, :]

        mri_slices_segmented.append(segmented_tumor)
    
        if display:
            img_utils.display_seg_results(currentSlice, slice_subs, binary_slice, tumor_mask, tumor_mask_final, segmented_tumor)

    if display:
        img_utils.plot_stack('Detected tumors', np.array(mri_slices_segmented))

    return mri_slices_segmented

def __mixture_of_gaussians(currentSlice):
    noOfGaussians = 3

    currentSliceVector = np.array(currentSlice.flatten())
    sliceTransposed = np.array([currentSliceVector])
    sliceTransposed = sliceTransposed.T

    gm = GaussianMixture(n_components=noOfGaussians, random_state=0, n_init=3).fit(sliceTransposed)
    gm_means = np.array(gm.means_).flatten()
    gm_covariances = np.array(gm.covariances_).flatten()
    
    maxV = np.max(gm_covariances)
    maxInd = np.where(gm_covariances == maxV)
    maxMu = max(gm_means[maxInd])

    return maxMu

def __morph_reconstruction(currentSlice, maxMu):
    seedImage = currentSlice - maxMu
    slice_rec = morphology.reconstruction(seedImage, currentSlice)
    slice_subs = currentSlice - slice_rec # get local maximas

    return slice_subs

def __binary_thresholding(currentSlice, slice_subs, maxMu):
    percentil = 99.9
    SubstractConst = np.percentile(currentSlice, percentil) - maxMu
    binary_slice = slice_subs > SubstractConst

    return binary_slice

def __create_tumor_mask(binary_slice, slice_subs):
    binary_slice_dil = morphology.dilation(binary_slice, morphology.square(4))
    binary_slice_er = morphology.erosion(binary_slice_dil, morphology.square(3))

    labels = measure.label(binary_slice_er)
    regions = measure.regionprops(labels)
    
    idxMax = __select_region(slice_subs, labels, regions)

    tumor_mask = np.ndarray([binary_slice.shape[0], binary_slice.shape[1]], dtype=np.float)
    tumor_mask[:] = 0
    if len(regions) > 0:
        tumor_mask = tumor_mask + np.where(labels == regions[idxMax].label, 1, 0)

    return tumor_mask

def __select_region(slice_subs, labels, regions):
    meanMax = 0
    idxMax = 0
    for region_idx in range(len(regions)):
        locs = np.where(labels == (region_idx+1))
        if regions[region_idx].area > 250: # set minimum area
            pixels = slice_subs[locs]
            region_mean = np.mean(pixels)
            if region_mean > meanMax:
                idxMax = region_idx
                meanMax = region_mean

    return idxMax

def __final_morph_op(tumor_mask):
    tumor_mask = morphology.dilation(tumor_mask, morphology.square(4))
    tumor_mask = morphology.closing(tumor_mask, morphology.square(4))
    tumor_mask = ndimage.binary_fill_holes(tumor_mask)
    tumor_mask = morphology.erosion(tumor_mask, morphology.square(1))

    return tumor_mask