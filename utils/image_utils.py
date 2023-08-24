from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

import numpy as np


def load_img_and_preprocess(path, shape=None):
    """
    Load and preprocess an image from a file.

    :param str path: Path to the image file.
    :param tuple shape: Desired shape of the image (height, width)
    :return: Preprocessed image array and original image array
    :rtype: tuple (numpy.ndarray, numpy.ndarray)
    """
    try:
        if shape is not None:
            if len(shape) != 2:
                raise ValueError("Shape must be a tuple of length 2 (height, width)")
            img = image.load_img(path, target_size=shape)
        else:
            img = image.load_img(path)
    except (FileNotFoundError, IOError):
        raise FileNotFoundError("Unable to open image file at path: {}".format(path))
    x_arr = image.img_to_array(img)
    x_expanded = np.expand_dims(x_arr, axis=0)
    x = preprocess_input(x_expanded)

    return x, x_arr


def scale_img(x):
    """
    Scale the pixel values of an image to the range [0, 1]

    :param numpy.ndarray x : Input image array
    :return: Scaled image array with pixel values in the range [0, 1]
    :rtype: numpy.ndarray
    """
    if isinstance(x, np.ndarray):
        if x.ndim != 3:
            raise ValueError("Input must be a 3D array representing an image")

        if not np.issubdtype(x.dtype, np.number):
            raise ValueError("Input array must contain numeric values")

        if x.size > 0:
            min_val = np.min(x)
            max_val = np.max(x)

            if min_val != max_val:
                scaled_x = (x - min_val) / (max_val - min_val)
            else:
                scaled_x = x * 0.0  # Avoid division by zero
        else:
            scaled_x = x  # Empty array, no scaling needed
    else:
        raise ValueError("`x` must be a numpy array")

    return scaled_x


def preprocess_reverse(img):
    """
    Reverse the preprocessing steps applied to an image.

    :param numpy.ndarray img: Preprocessed image array
    :return: Reversed preprocessed image array
    :rtype: numpy.ndarray
    """
    RED_CHANNEL_MEAN = 103.939
    GREEN_CHANNEL_MEAN = 116.779
    BLUE_CHANNEL_MEAN = 126.68

    if img.size == 0:
        raise ValueError("Input image array is empty")
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")
    if img.ndim != 4:
        raise ValueError("img must have 4 dimensions")

    img_copy = img.copy()
    img_copy[..., 0] += RED_CHANNEL_MEAN
    img_copy[..., 1] += GREEN_CHANNEL_MEAN
    img_copy[..., 2] += BLUE_CHANNEL_MEAN
    img_copy = img_copy[..., ::-1]    # Reverses the order of color channels in the image
    # In many deep learning frameworks, images are often loaded in RGB (Red-Green-Blue) channel order,
    # but certain models, like VGG16, were pre-trained with images in BGR (Blue-Green-Red) channel order.
    # Therefore, when preprocessing images for these models, the channels need to be reversed from RGB to BGR.

    return img_copy