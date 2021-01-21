import matplotlib.pyplot as plt
import numpy as np


def plot_hist(subject, imgs):
    plt.hist(imgs.flatten(), bins=256, color='c')
    plt.xlabel("Voxel values")
    plt.ylabel("Frequency")
    plt.title(subject)
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


def plot_stack(subject, img, rows=6, cols=4, start_with=0):
    show_every = int(img.shape[0]/(rows*cols))
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
    plt.show()


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
    input /= np.max(input) / (max - min)
    input += min
    return input
