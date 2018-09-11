"""
Utils for working with CamVid images.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def normalize_image(images):
    """Normalize CamVid images to mean 0.0 and std 1.0. The method works 
       for a single image or a batch of images. The result is an array of
       type float32 with the same shape as the input.
    """
    images = images / 255.
    images -= 0.4 # approx. mean sampled from the dataset 
    images /= 0.3 # approx. std sampled from the dataset
    return images    

def unnormalize_image(images):
    """Reverse the normalization of `normalize_image()`. This is mostly 
       useful for ploting. The method works for a single image or a batch 
       of images. The result is an array of type uint8 with the same shape 
       as the input.
    """
    img = (images * 0.3 + 0.4) * 255.
    img = np.clip(img, 0.0, 255.0)
    img = img.astype(np.uint8)
    return img

def load_label_colors(label_colors_path):
    """Load the `label_colors.txt` file from the CamVid dataset.
       Return:
           label_codes: list of label codes, arrays with RGB values
           label_names: list of label code names
           label_code2id: dict that maps label codes to unique IDs
    """
    def parse_code(l):
        a, b = l.strip().split("\t")
        return tuple(int(o) for o in a.split(' ')), b

    label_codes, label_names = zip(*[parse_code(l) for l in open(label_colors_path)])
    label_codes = list(label_codes)
    label_names = list(label_names)

    # assign IDs to codes
    label_code2id = {v:k for k,v in enumerate(label_codes)} 
    
    return label_codes, label_names, label_code2id

def color_label(label, label_colors):
    """Map label IDs to colors.
       Args: 
           label: array with shape [height, width]
           label_colors: list of RGB colors
       Return:
           array with shape [height, width, 3]
    """
    assert len(label.shape) == 2
    r, c = label.shape
    res = np.zeros((r,c,3), 'uint8')
    for j in range(r): 
        for k in range(c):
            o = label_colors[label[j,k]]
            res[j,k] = o
    return res

def show_image_row(images, figsize=(15, 15), show_axis=True):
    fig = plt.figure(figsize=figsize)
    number_of_images = len(images)
    for i in range(number_of_images):
        a = fig.add_subplot(1, number_of_images, i+1)
        plt.imshow(images[i])
        if not show_axis:
            plt.axis('off')