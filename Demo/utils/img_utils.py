import matplotlib.pyplot as plt
import numpy as np
from skimage import measure


def plot_hist(subject, imgs):
    plt.hist(imgs.flatten(), bins=256, color='c')
    plt.xlabel("Voxel values")
    plt.ylabel("Frequency")
    plt.title(subject)
    plt.show()


def plot_hist_peaks(subject, hist, peaks):
    plt.plot(hist)
    plt.xlabel("Voxel values")
    plt.ylabel("Frequency")
    plt.title(subject)
    plt.plot(peaks, hist[peaks], 'x')
    plt.show()


def plot_hist_thresholds(img, thresholds):
    # Using the threshold values, we generate the three regions.
    regions = np.digitize(img, bins=thresholds)
    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

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


def plot_stack(subject, img, rows=6, cols=4, start_with=0, block=True):
    show_every = round(img.shape[0]/(rows*cols))
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        if rows != 1:
            if ind < img.shape[0]:
                ax[int(i/cols), int(i % cols)].set_title('slice %d' %
                                                         ind, fontsize=7)
                ax[int(i/cols), int(i % cols)
                   ].imshow(img[ind, :, :], cmap='gray')

            ax[int(i/cols), int(i % cols)].axis('off')
        else:
            if ind < img.shape[0]:
                ax[ind].set_title('slice %d' % ind, fontsize=7)
                ax[ind].imshow(img[ind, :, :], cmap='gray')

            ax[ind].axis('off')

    fig.suptitle(subject)
    if block:
        plt.show()
    else:
        plt.show(block=block)
        plt.pause(10)
        plt.close()


def save_stack_contours(fig_name, s, mri_slices_preprocessed,
                        mri_slices_masks, rows=6, cols=4, start_with=0):

    show_every = round(mri_slices_preprocessed.shape[0]/(rows*cols))
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        if rows != 1:
            if ind < mri_slices_preprocessed.shape[0]:
                ax[int(i/cols), int(i % cols)].set_title('slice %d' %
                                                         ind, fontsize=7)
                ax[int(i/cols), int(i % cols)
                   ].imshow(mri_slices_preprocessed[ind, :, :], cmap='gray')

                contours = measure.find_contours(mri_slices_masks[ind, :, :])
                if len(contours):
                    ax[int(i/cols), int(i % cols)].scatter(x=contours[0]
                                                           [:, 1], y=contours[0][:, 0], c='r', s=1)

            ax[int(i/cols), int(i % cols)].axis('off')
        else:
            if ind < mri_slices_preprocessed.shape[0]:
                ax[ind].set_title('slice %d' % ind, fontsize=7)
                ax[ind].imshow(mri_slices_preprocessed[ind, :, :], cmap='gray')

                contours = measure.find_contours(mri_slices_masks[ind, :, :])
                if len(contours):
                    ax[int(i/cols), int(i % cols)].scatter(x=contours[0]
                                                           [:, 1], y=contours[0][:, 0], c='r', s=1)

            ax[ind].axis('off')

    fig.suptitle(s)
    plt.savefig(fig_name, bbox_inches='tight')


def plot_stack_documentation_img(subject, img, titles, rows=6, cols=4, start_with=0):
    show_every = int(img.shape[0]/(rows*cols))
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        if rows != 1:
            if ind < img.shape[0]:
                ax[int(i/cols), int(i % cols)
                   ].set_title(titles[ind], fontsize=7)
                ax[int(i/cols), int(i % cols)
                   ].imshow(img[ind, :, :], cmap='gray')

            ax[int(i/cols), int(i % cols)].axis('off')
        else:
            if ind < img.shape[0]:
                ax[int(i % cols)].set_title(titles[ind], fontsize=7)
                ax[int(i % cols)].imshow(img[ind, :, :], cmap='gray')

            ax[int(i % cols)].axis('off')

    fig.suptitle(subject)
    plt.show()


def plot_img(subject, img):
    plt.imshow(img, cmap='gray')
    plt.title(subject)
    plt.axis('off')
    plt.show()


def scale_range(input, min, max):
    input += -(np.min(input))
    input = input / (np.max(input) / (max - min))
    input += min
    return input


def display_seg_results(currentSlice, slice_subs, binary_slice, tumor_mask, tumor_mask_final, segmented_tumor):
    _, ax = plt.subplots(nrows=2, ncols=3, figsize=(40, 3.5))

    ax[0, 0].imshow(currentSlice, cmap='gray')
    ax[0, 0].set_title('Original')
    ax[0, 0].axis('off')

    ax[0, 1].imshow(slice_subs, cmap='gray')
    ax[0, 1].set_title('Morphological reconstruction')
    ax[0, 1].axis('off')

    ax[0, 2].imshow(binary_slice, cmap='gray')
    ax[0, 2].set_title('Binary thresholding')
    ax[0, 2].axis('off')

    ax[1, 0].imshow(tumor_mask, cmap='gray')
    ax[1, 0].set_title('Tumor region')
    ax[1, 0].axis('off')

    ax[1, 1].imshow(tumor_mask_final, cmap='gray')
    ax[1, 1].set_title('Final tumor mask')
    ax[1, 1].axis('off')

    ax[1, 2].imshow(segmented_tumor, cmap='gray')
    ax[1, 2].set_title('Segmented tumor')
    ax[1, 2].axis('off')

    plt.show()
