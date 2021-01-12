import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from skimage import measure
from scipy import ndimage


def contrast_enhancement(raw_slices):
    pass


def get_brain_mask(slc, display):
    # Threshold binary image
    threshold = np.std(slc)
    thresh_slc = np.where(slc < threshold, 0.0, 1.0)  # threshold the image

    # Erode binary image
    eroded = morphology.erosion(thresh_slc, np.ones([3, 3]))

    # Extract 2 largest BLOB
    object_labels = measure.label(eroded)
    unique_labels = np.unique(object_labels)
    regions = measure.regionprops(object_labels)
    regions.sort(key=lambda region: region.area, reverse=True)
    # Extract the largest BLOB
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
        fig, ax = plt.subplots(2, 3, figsize=[12, 12])
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


def image_preprocessing(raw_slices):
    contrast_enhancement(raw_slices)
    no_skull_slices = skull_strip(raw_slices)

    return no_skull_slices


def plot_hist(subject, imgs):
    plt.hist(imgs.flatten(), bins=50, color='c')
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
