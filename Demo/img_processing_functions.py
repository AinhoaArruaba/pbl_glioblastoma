import numpy as np
from skimage import morphology
from skimage import measure
from skimage import filters
from skimage import exposure

from medpy.filter import smoothing

from scipy import ndimage

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


def contrast_enhancement(raw_slices):
    enhanced_slices = np.array(
        [exposure.equalize_adapthist(s) for s in raw_slices])
    return enhanced_slices

# S3 Skull stripping algorithm implementation for 2D brain MRI


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


def skull_strip_s3_algorithm(raw_slices, display):
    no_skull_slices = []
    for index in range(raw_slices.shape[0]):
        mask = get_brain_mask_s3(raw_slices[index, :, :], display)
        no_skull_slices.append(mask*raw_slices[index, :, :])

    return no_skull_slices

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
        plot_stack('', np.array(coarse_masks))

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
        # plot_hist_thresholds(raw_slices[14, :, :], thresholds)
        # plot_stack('', np.array(lv1_filtered_slices))
        plot_stack('Thresh mask', np.array(thresh_masks))

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

    for r in range(4):
        bse_masks = morphology.dilation(bse_masks, morphology.cube(1))
        bse_masks = ndimage.binary_fill_holes(bse_masks)
        bse_masks = morphology.closing(bse_masks, morphology.ball(4))
        bse_masks = morphology.erosion(bse_masks, morphology.cube(1))

    for r in range(2):
        for index in range(bse_masks.shape[0]):
            bse_masks[index, :, :] = ndimage.binary_fill_holes(
                bse_masks[index, :, :])

    if display:
        plot_stack('BSE mask', bse_masks)

    return bse_masks


def get_consensus_mask(coarse_masks, thresh_masks, bse_masks, display=False):
    consensus_mask = coarse_masks
    for index in range(coarse_masks.shape[0]):
        for row in range(coarse_masks.shape[1]):
            for col in range(coarse_masks.shape[2]):
                if (coarse_masks[index, row, col] and thresh_masks[index, row, col]) or (coarse_masks[index, row, col] and bse_masks[index, row, col]) or (thresh_masks[index, row, col] and bse_masks[index, row, col]):
                    consensus_mask[index, row, col] = True
                else:
                    consensus_mask[index, row, col] = False

    if display:
        plot_stack('Consensus mask', consensus_mask)

    return consensus_mask


def skull_strip_mcstrip_algorithm(raw_slices, display=False):
    coarse_masks = get_brain_coarse_mask(raw_slices, display=False)
    thresh_masks = get_brain_thresh_mask(
        raw_slices, coarse_masks, display=False)
    bse_masks = get_brain_bse_mask(
        raw_slices, coarse_masks, thresh_masks, display=True)
    # brain_masks = get_consensus_mask(
    #     coarse_masks, thresh_masks, bse_masks, display=False)

    no_skull_slices = []
    for index in range(raw_slices.shape[0]):
        no_skull_slices.append(
            bse_masks[index, :, :]*raw_slices[index, :, :])

    if display:
        plot_stack('Masked brain', np.array(no_skull_slices))

    return no_skull_slices


def image_preprocessing(raw_slices, dicom_data, subject, display):
    # enhanced_slices = contrast_enhancement(raw_slices)

    no_skull_slices = skull_strip_mcstrip_algorithm(raw_slices, display)

    return no_skull_slices


def plot_hist(subject, imgs):
    plt.hist(imgs.flatten(), bins=256, color='c')
    plt.xlabel("Voxel values")
    plt.ylabel("Frequency")
    plt.title(subject)
    plt.show()


def plot_hist_thresholds(img, thresholds):
    # Using the threshold values, we generate the three regions.
    regions = np.digitize(img, bins=thresholds)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].hist(img.ravel(), bins=255)
    ax[1].set_title('Histogram')
    for thresh in thresholds:
        ax[1].axvline(thresh, color='r')

    ax[2].imshow(regions, cmap='jet')
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')
    plt.show()


def plot_stack(subject, img, rows=6, cols=4, start_with=0, show_every=7):
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        if ind < img.shape[0]:
            ax[int(i/cols), int(i % cols)].set_title('slice %d' %
                                                     ind, fontsize=7)
            ax[int(i/cols), int(i % cols)].imshow(img[ind, :, :], cmap='gray')
        ax[int(i/cols), int(i % cols)].axis('off')

    fig.suptitle(subject)
    plt.show()


def plot_img(subject, img):
    plt.imshow(img, cmap='gray')
    plt.title(subject)
    plt.axis('off')
    plt.show()


def make_mesh(image, threshold=-300, step_size=1):
    p = image.transpose(2, 1, 0)
    verts, faces, norm, val = measure.marching_cubes(
        p, step_size=step_size, allow_degenerate=True)
    return verts, faces


def plot_3d(image, threshold=-300, step_size=1):
    verts, faces = make_mesh(image)
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    # ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    plt.show()
