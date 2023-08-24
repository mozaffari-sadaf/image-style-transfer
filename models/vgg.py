from keras.layers import AveragePooling2D, MaxPooling2D
from keras.models import Sequential
from keras.applications.vgg16 import VGG16


def vgg_avg_pool(height=224, width=224, channels=3):
    """
    Create a VGG16 model with average pooling and custom input shape.

    :param int height: Height of the input image
    :param int width: Width of the input image
    :param int channels: Channels of the input image
    :return: A modified VGG16 model with average pooling layers.
    :rtype: keras.models.Model
    """

    # Adjust the input shape if it's too small
    min_input_size = 32
    if height is None or height < min_input_size:
        height = min_input_size

    if width is None or width < min_input_size:
        width = min_input_size

    vgg = VGG16(input_shape=(height, width, channels), weights='imagenet', include_top=False)
    # We want to initialize with weights pre-trained on the ImageNet dataset.
    # We set the include_top to False because want to use the model for feature extraction rather than classification

    new_model = Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            new_model.add(AveragePooling2D())
        else:
            new_model.add(layer)

    return new_model