import tensorflow as tf
import keras.backend as K


def gram_matrix(img):
    """
    Compute the Gram matrix of a given image.

    :param tensorflow.Tensor img: Input tensor representing a feature map
    :return: The computed Gram matrix
    :rtype: tensorflow.Tensor
    """
    # Input validation and casting
    if not isinstance(img, tf.Tensor):
        raise TypeError("Input 'img' must be a valid tensor.")

    if tf.size(img) == 0:
        raise ValueError("Input tensor 'img' is empty.")

    if len(img.shape) != 3:
        raise ValueError("Input tensor `img` must have 3 dimensions.")

    if img.dtype != tf.float32:
        img = tf.cast(img, tf.float32)

    # Compute Gram matrix
    X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
    G = K.dot(X, K.transpose(X)) / tf.size(img, out_type=tf.float32)

    return G


def style_loss(y, t):
    """
    Compute the style loss between two feature maps.

    :param tensorflow.Tensor y: First input tensor representing a feature map.
    :param tensorflow.Tensor t: Second input tensor representing a feature map.
    :return: Computed style loss.
    :rtype: tensorflow.Tensor
    """
    try:
        # Check if y and t are empty tensors
        if tf.size(y) == 0 or tf.size(t) == 0:
            raise ValueError("Input tensors 'y' and 't' cannot be empty.")

        # Check dimensions of y and t
        if y.shape != t.shape:
            raise ValueError("Input tensors 'y' and 't' must have the same dimensions.")

        # Check if y and t have the same data type
        if y.dtype != t.dtype:
            raise ValueError("Input tensors 'y' and 't' must have the same data type.")

        # Calculate style loss using Gram matrices
        return tf.reduce_mean(K.square(gram_matrix(y) - gram_matrix(t)))
    except Exception as e:
        raise ValueError("Error calculating style loss: " + str(e))