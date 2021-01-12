import matplotlib.pyplot as plt


def contrast_enhancement(raw_slices):
    pass


def skull_strip(raw_slices):
    no_skull_slices = raw_slices
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
