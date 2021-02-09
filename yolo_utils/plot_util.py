import matplotlib.pyplot as plt
import numpy as np

def show_grid(images, max_rows=3, max_cols=3, figsize=(20,20), show_axis='off'):
    '''
    images : np.ndarray (N, h, w, chnl)
    N >= max_rows * max_cols 
    '''
    fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=figsize)

#     for r in range(max_rows):
#         for c in range(max_cols):
#             axes[r, c].axis('off')
    
    for idx, image in enumerate(images):
        row = idx // max_cols
        col = idx % max_cols
        axes[row, col].axis(show_axis)
        axes[row, col].imshow(image, cmap="gray", aspect="auto")
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show()

def show_batch(imgs, figsize=(20,20), axis="off", titles=[]):
    '''Example:
    show_batch(augmented_imgs,
          titles=[img.shape for img in augmented_imgs])
    '''
    plt.figure(figsize=figsize)
    img_l = len(imgs)
    sqrt = np.sqrt(img_l)
    row = np.ceil(sqrt).astype('int')
    col = np.floor(sqrt).astype('int')

    show_title = True if len(titles) == img_l else False
    
    for i in range(img_l):
        ax = plt.subplot(row, col, i + 1)
        plt.imshow(imgs[i])
        if show_title:
            plt.title(titles[i])

    plt.axis(axis)


