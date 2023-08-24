# Standard Library Imports
from __future__ import print_function, division

import argparse
from builtins import range
from datetime import datetime

# Third-Party Library Imports
import tensorflow as tf
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from keras.models import Model  # Import Model class from Keras

# Custom Module Imports
from models.vgg import vgg_avg_pool  # Import your custom VGG model
from utils.image_utils import load_img_and_preprocess, scale_img, preprocess_reverse
from utils.style_utils import style_loss

# Disable eager execution for TensorFlow 2.x compatibility
tf.compat.v1.disable_eager_execution()

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Image Style Transfer")
parser.add_argument("--content", required=True, help="Content image file path")
parser.add_argument("--style", required=True, help="Style image file path")
args = parser.parse_args()

# Constants to replace magic numbers
CONTENT_WEIGHT = 1  # Weight for content loss in the overall loss function
STYLE_WEIGHTS = [1, 2, 3, 4]    # Style weights to balance the importance of different style layers
OPTIMIZATION_ITERATIONS = 10    # Number of iterations for the optimization process

# Optimization parameters
MAX_ITERATIONS = 20  # Number of optimization iterations
CLIP_MIN = -127
CLIP_MAX = 127

# Load and preprocess the content image
content_img, content = load_img_and_preprocess('./images/' + args.content,)

# Get the height and width of the content image
h, w = content_img.shape[1:3]

# Load and preprocess the style image, matching the dimensions of the content image
style_img, style = load_img_and_preprocess('./images/' + args.style, (h, w))

# Get the batch shape based on the content image
batch_shape = content_img.shape
shape = content_img.shape[1:]

# Defining VGG model and layers
vgg = vgg_avg_pool(shape[0], shape[1], shape[2])

# Selecting a content layer for capturing content features
content_model = Model(vgg.input, vgg.layers[7].output)  # Layer 7 captures intermediate features

# Predict content features using the content model and content image
content_target = K.variable(content_model.predict(content_img))

# Selecting convolutional layers for capturing style features
symbolic_conv_outputs = [layer.output for layer in vgg.layers if layer.name.endswith('conv1')]

# Creating a style model to capture style features
style_model = Model(vgg.input, symbolic_conv_outputs)

# Extracting style features from the style image
style_outputs = [K.variable(y) for y in style_model.predict(style_img)]

# Calculating content loss
loss = CONTENT_WEIGHT * K.mean(K.square(content_model.output - content_target))

# Calculating style loss and adding it to the total loss
for w, symbolic, actual in zip(STYLE_WEIGHTS, symbolic_conv_outputs, style_outputs):
    loss += w * style_loss(symbolic[0], actual[0])

# Computing gradients
gradients = K.gradients(loss, vgg.input)

# Defining the function to compute loss and gradients
get_loss_and_gradients = K.function(
    inputs=[vgg.input],
    outputs=[loss]+gradients
)


# Wrapper function for get_loss_and_grads
def get_loss_and_gradients_wrapper(x_vector):
    """
    Calculate loss and gradients using the wrapped get_loss_and_grads function.

    Args:
        x_vector (ndarray): Input vector.

    Returns:
        tuple: Computed loss and flattened gradients.
    """

    if not isinstance(x_vector, np.ndarray):
        raise TypeError("x_vector must be a numpy array")

    try:
        # Calculate loss and gradients using the wrapped get_loss_and_grads function
        loss_value, gradients_value = get_loss_and_gradients([x_vector.reshape(*batch_shape)])

        # Return the computed loss and flattened gradients
        return loss_value.astype(np.float64), gradients_value.flatten().astype(np.float64)
    except Exception as e:
        # Handle the exception and return an appropriate error message or take necessary actions
        print("Error occurred during execution of get_loss_and_grads:", str(e))
        return None, None


def minimize_loss(func, epochs, batch_shape):
    """
    Performs L-BFGS optimization to minimize a given loss function and
    generates a stylized image by iterating over multiple epochs and optimizing
    the image to minimize the loss.

    :param function func: The loss and gradient function to be minimized
    :param int epochs:  The number of optimization iterations
    :param tuple batch_shape: The shape of the input image batch
    :return: The stylized image generated by the optimization process
    :rtype: numpy.ndarray
    """
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))

    for i in range(epochs):
        # Perform the L-BFGS optimization
        x, l, _ = fmin_l_bfgs_b(
            func=func,
            x0=x,
            maxfun=20
        )

        x = np.clip(x, -127, 127)

        print("ite=%s, loss=%s" % (i, l))
        losses.append(l)

    # Print the duration of optimization
    print("duration:", datetime.now() - t0)
    plt.plot(losses)
    plt.show()

    # Reshape the optimized image and reverse preprocessing
    generated_img = x.reshape(*batch_shape)
    stylized_img = preprocess_reverse(generated_img)

    return stylized_img[0]


# Minimizing loss anf generating stylized image
stylized_img = minimize_loss(get_loss_and_gradients_wrapper, OPTIMIZATION_ITERATIONS, batch_shape)

# Displaying the final stylized image
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(scale_img(content))
plt.title("Content Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(scale_img(stylized_img))
plt.title("Stylized Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(scale_img(style))
plt.title("Style Image")
plt.axis("off")

plt.tight_layout()
plt.show()