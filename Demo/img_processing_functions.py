import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from skimage import measure
from skimage import filters
from sklearn.cluster import KMeans

from scipy import ndimage


def contrast_enhancement(raw_slices):
    pass


"""
def get_binary_slc(slc):
    # fig, ax = filters.try_all_threshold(slc, figsize=(10, 8), verbose=False)
    # plt.show()
    # Thresholding with Otsu's method
    # threshold = filters.threshold_otsu(slc)

    # Thresholding method KMeans
    row_size = slc.shape[0]
    col_size = slc.shape[1]
    mean = np.mean(slc)
    std = np.std(slc)
    slc = slc-mean
    slc = slc/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = slc[int(col_size*(1/5)):int(col_size*(4/5)),
                 int(row_size*(1/5)):int(row_size*(4/5))]
    mean = np.mean(middle)
    max = np.max(slc)
    min = np.min(slc)
    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    slc[slc == max] = mean
    slc[slc == min] = mean
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    kmeans = KMeans(n_clusters=2).fit(
        np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)

    # Apply threshold to obtain binary slice
    bin_slc = slc > threshold

    return bin_slc

def get_brain_mask(slc, display):
    row_size = slc.shape[0]
    col_size = slc.shape[1]
    thresh_slc = get_binary_slc(slc)
    # Erode binary image
    eroded = morphology.erosion(thresh_slc, np.ones([5, 5]))
    # Extract 2 largest BLOB
    object_labels = measure.label(eroded)
    labels = []
    # unique_labels = np.unique(object_labels)
    regions = measure.regionprops(object_labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    # bbox values -> (min_row, min_col, max_row, max_col)
    # for region in regions:
    #     region_bbox = region.bbox
    #     if region_bbox[3]-region_bbox[1] < col_size*(8.5/10) and region_bbox[3] < col_size*(8.5/10):
    #         labels.append(region.label)
    # mask = np.ndarray([row_size, col_size], dtype=np.int8)
    # mask[:] = 0
    # for label in labels:
    #     mask = mask + np.where(object_labels == label, 1, 0)

    # # Extract the largest BLOB
    mask = np.ndarray([slc.shape[0], slc.shape[1]], dtype=np.int16)
    mask[:] = 0
    mask = mask + np.where(object_labels == regions[0].label, 1, 0)
    # Erode binary image
    mask_eroded = morphology.erosion(mask, np.ones([3, 3]))
    # Fill binary image
    mask_filled = ndimage.binary_fill_holes(mask_eroded)
    # Dilate the binary image
    mask_final = morphology.dilation(mask_filled, np.ones([8, 8]))
    # mask_final = np.logical_not(mask_final).astype(int)
    if display:
        fig, ax = plt.subplots(2, 3, figsize=[9, 9])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(slc, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_slc, cmap='gray')
        ax[0, 1].axis('off')
        ax[0, 2].set_title("After Erosion")
        ax[0, 2].imshow(eroded, cmap='gray')
        ax[0, 2].axis('off')
        ax[1, 0].set_title("Color Labels")
        ax[1, 0].imshow(object_labels)
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Final Mask")
        ax[1, 1].imshow(mask_final, cmap='gray')
        ax[1, 1].axis('off')
        ax[1, 2].set_title("Apply Mask on Original")
        ax[1, 2].imshow(mask_final*slc, cmap='gray')
        ax[1, 2].axis('off')

        plt.show()

    return 1

def skull_strip(raw_slices):
    no_skull_slices = []
    for index in range(raw_slices.shape[2]):
        mask = get_brain_mask(raw_slices[:, :, index], True)
        no_skull_slices.append(mask*raw_slices[:, :, index])

    return no_skull_slices
"""


def get_brain_mask_s3(img, display=False):
    # Algorithm: A Simple Skull Stripping Algorithm for Brain MRI
    # 10.1109/ICAPR.2015.7050671
    # 1) Apply median filtering with a window of size 3 × 3 to the input image.
    img_median = filters.median(img, morphology.disk(3))
    # 2) Compute the initial mean intensity value Ti of the image.
    mean_ti = np.mean(img_median)
    # 3) Identify the top, bottom, left, and right pixel locations, from where
    # brain skull starts in the image, considering gray values of the skull are greater than Ti.
    start_row = 0
    start_col = 0
    end_row = img_median.shape[0]
    end_col = img_median.shape[1]
    for index in range(img_median.shape[0]):
        if np.max(img_median[index, :]) > mean_ti:
            start_row = index
            break
    for index in reversed(range(img_median.shape[0])):
        if np.max(img_median[index, :]) > mean_ti:
            end_row = index
            break
    for index in range(img_median.shape[1]):
        if np.max(img_median[:, index]) > mean_ti:
            start_col = index
            break
    for index in reversed(range(img_median.shape[1])):
        if np.max(img_median[:, index]) > mean_ti:
            end_col = index
            break
    # 4) Form a rectangle using the top, bottom, left, and right pixel locations.
    # 5) Compute the final mean value Tf of the brain using the pixels located within
    # the rectangle.
    mean_tf = np.mean(img_median[start_row:end_row, start_col:end_col]
                      [img_median[start_row:end_row, start_col:end_col] > 0])
    # 6) Approximate the region of brain membrane or meninges that envelop the brain,
    # based on the assumption that the intensity of skull is more than Tf and that of
    # membrane is less than Tf.
    # 7) Set the average intensity value of membrane as the threshold value T.
    mask = np.logical_and((img_median[start_row:end_row, start_col:end_col] < mean_tf), (
        img_median[start_row:end_row, start_col:end_col] > 0))
    t = np.mean(img_median[start_row:end_row, start_col:end_col][mask])
    # 8) Convert the given input image into binary image using the threshold T.
    binary_img = img > mean_tf
    # 9) Apply a 13 × 13 opening morphological operation to the binary image in order to
    # separate the skull from the brain completely.
    opening_img = morphology.dilation(binary_img, morphology.square(2))
    opening_img = morphology.opening(opening_img, morphology.square(13))
    opening_img = morphology.dilation(opening_img, morphology.square(2))
    # 10) Find the largest connected component and consider it as brain.
    labels = measure.label(opening_img)
    regions = measure.regionprops(labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    brain_mask = np.ndarray([img.shape[0], img.shape[1]], dtype=np.int8)
    brain_mask[:] = 0
    brain_mask = brain_mask + np.where(labels == regions[0].label, 1, 0)
    # 11) Finally, apply a 21 × 21 closing morphological operation to fill the gaps
    # within and along the periphery of the intracranial region.
    brain_mask = morphology.closing(brain_mask, morphology.square(21))
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    brain_mask = morphology.dilation(brain_mask, morphology.square(5))

    if display:
        fig, ax = plt.subplots(2, 3, figsize=[9, 9])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Median filtered")
        ax[0, 1].imshow(img_median, cmap='gray')
        ax[0, 1].axis('off')
        ax[0, 2].set_title("Membrane img")
        ax[0, 2].imshow(binary_img, cmap='gray')
        ax[0, 2].axis('off')
        ax[1, 0].set_title("Opening morphological operation")
        ax[1, 0].imshow(opening_img)
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Selected region Mask")
        ax[1, 1].imshow(brain_mask, cmap='gray')
        ax[1, 1].axis('off')
        ax[1, 2].set_title("Apply Mask on Original")
        ax[1, 2].imshow(brain_mask*img, cmap='gray')
        ax[1, 2].axis('off')

        plt.show()
    return 1


def skull_strip_s3(raw_slices, display):
    no_skull_slices = []
    for index in range(raw_slices.shape[2]):
        mask = get_brain_mask_s3(raw_slices[:, :, index], display)
        no_skull_slices.append(mask*raw_slices[:, :, index])

    return no_skull_slices


def image_preprocessing(raw_slices, display):
    contrast_enhancement(raw_slices)
    no_skull_slices = skull_strip_s3(raw_slices, display)

    return no_skull_slices


def plot_hist(subject, imgs):
    plt.hist(imgs.flatten(), bins=256, color='c')
    plt.xlabel("Voxel values")
    plt.ylabel("Frequency")
    plt.title(subject)
    plt.show()


def plot_stack(subject, img, rows=6, cols=4, start_with=0, show_every=1):
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        if ind < img.shape[2]:
            ax[int(i/cols), int(i % cols)].set_title('slice %d' %
                                                     ind, fontsize=7)
            ax[int(i/cols), int(i % cols)].imshow(img[:, :, ind], cmap='gray')
        ax[int(i/cols), int(i % cols)].axis('off')

    fig.suptitle(subject)
    plt.show()


def plot_img(subject, img):
    plt.imshow(img, cmap='gray')
    plt.title(subject)
    plt.axis('off')
    plt.show()
