import unittest

import pytest

from models.vgg import vgg_avg_pool
from utils.style_utils import gram_matrix, style_loss
from utils.image_utils import load_img_and_preprocess, scale_img, preprocess_reverse

import tensorflow as tf
import keras.backend as K
import numpy as np


class TestVGG(unittest.TestCase):

    # Tests that the function creates a VGG16 model with default input shape.
    def test_default_input_shape(self):
        model = vgg_avg_pool(224, 224, 3)
        self.assertEqual(model.input_shape, (None, 224, 224, 3))

    # Tests that the function creates a VGG16 model with custom input shape.
    def test_custom_input_shape(self):
        model = vgg_avg_pool(128, 128, 3)
        self.assertEqual(model.input_shape, (None, 128, 128, 3))

    # Tests that the function creates a VGG16 model with input shape of (1, 1, 3).
    def test_input_shape_1_1_3(self):
        model = vgg_avg_pool(1, 1, 3)
        self.assertEqual(model.input_shape, (None, 32, 32, 3))

    # Tests that the function creates a VGG16 model with input shape of (None, None, 3).
    def test_input_shape_none_none_3(self):
        model = vgg_avg_pool(None, None, 3)
        self.assertEqual(model.input_shape, (None, 32, 32, 3))


class TestGramMatrix(unittest.TestCase):

    # Tests that the function correctly computes the Gram matrix for a 3D tensor of shape (3, 3, 3) with random values.
    def test_3d_tensor_shape_3_3_3(self):
        # Create a 3D tensor of shape (3, 3, 3) with random values
        img = tf.random.normal((3, 3, 3))

        # Compute the expected Gram matrix
        X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
        expected_G = K.dot(X, K.transpose(X)) / tf.size(img, out_type=tf.float32)

        # Compute the actual Gram matrix using the gram_matrix function
        actual_G = gram_matrix(img)

        # Assert that the actual Gram matrix is equal to the expected Gram matrix
        assert K.all(K.equal(actual_G, expected_G))

    # Tests that the function correctly computes the Gram matrix for a 3D tensor of shape (1, 3, 3) with random values.
    def test_3d_tensor_shape_1_3_3(self):
        # Create a 3D tensor of shape (1, 3, 3) with random values
        img = tf.random.normal((1, 3, 3))

        # Compute the expected Gram matrix
        X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
        expected_G = K.dot(X, K.transpose(X)) / tf.size(img, out_type=tf.float32)

        # Compute the actual Gram matrix using the gram_matrix function
        actual_G = gram_matrix(img)

        # Assert that the actual Gram matrix is equal to the expected Gram matrix
        assert K.all(K.equal(actual_G, expected_G))


class TestStyleLoss(unittest.TestCase):

    # Tests that style loss is computed correctly between two feature maps of the same dimensions.
    def test_compute_style_loss_same_dimensions(self):
        # Create two feature maps with the same dimensions
        y = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.float32)
        t = tf.constant([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]], dtype=tf.float32)

        # Compute the style loss
        loss = style_loss(y, t)

        # Check that the style loss is computed correctly
        assert loss == tf.reduce_mean(tf.square(gram_matrix(y) - gram_matrix(t)))

    # Tests that style loss is computed correctly with empty feature maps.
    def test_compute_style_loss_empty_feature_maps(self):
        # Create empty feature maps (tensors with no elements)
        empty_map = tf.constant([], dtype=tf.float32)

        # Assert that calling the style_loss function with empty maps raises a ValueError
        with self.assertRaises(ValueError):
            style_loss(empty_map, empty_map)

    # Tests that style loss is computed correctly between two feature maps of different data types.
    def test_compute_style_loss_valid_inputs(self):
        # Create two feature maps with valid inputs
        y = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.float32)
        t = tf.constant([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]], dtype=tf.int32)

        # Assert that calling the style_loss function with feature maps of different data types raises a ValueError
        with self.assertRaises(ValueError):
            style_loss(y, t)

    # Tests that style loss is computed correctly between two feature maps of different values.
    def test_compute_style_loss_different_values(self):
        # Create two feature maps with different values
        y = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.float32)
        t = tf.constant([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]], dtype=tf.float32)

        # Compute the style loss
        loss = style_loss(y, t)

        # Check that the style loss is computed correctly
        assert loss == tf.reduce_mean(tf.square(gram_matrix(y) - gram_matrix(t)))


class LoadImgAndPreprocess(unittest.TestCase):

    # Tests that the function can load and preprocess an image without specifying shape
    def test_load_and_preprocess_image_without_shape(self):
        # Arrange
        path = "../images/city.png"

        # Act
        result = load_img_and_preprocess(path)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)

    # Tests that the function can load and preprocess an image with a specified shape
    def test_load_and_preprocess_specified_shape(self):
        # Arrange
        path = "../images/city.png"
        shape = (300, 300)

        # Act
        result = load_img_and_preprocess(path, shape)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)

    # Tests that the function handles an invalid image path correctly
    def test_load_and_preprocess_invalid_path(self):
        # Arrange
        path = "invalid_image.jpg"
        shape = None

        # Act and Assert
        with pytest.raises(FileNotFoundError):
            load_img_and_preprocess(path, shape)

    # Tests that the function handles an invalid image shape correctly
    def test_load_and_preprocess_invalid_shape(self):
        # Arrange
        path = "../images/city.png"
        shape = (100, 100, 3)

        # Act and Assert
        with pytest.raises(ValueError):
            load_img_and_preprocess(path, shape)

    # Tests that the return type of the function is a tuple
    def test_return_type_is_tuple(self):
        # Arrange
        path = "../images/city.png"

        # Act
        result = load_img_and_preprocess(path)

        # Assert
        assert isinstance(result, tuple)


class TestScaleImg(unittest.TestCase):

    # Tests that the function scales an input array with all pixel values between 0 and 1 correctly
    def test_happy_path_all_pixel_values_between_0_and_1(self):
        # Create an input array with pixel values between 0 and 1
        input_array = np.array([[[0.2, 0.4, 0.6], [0.8, 0.1, 0.9], [0.4, 0.8, 0.5]]])

        # Call the function under test
        scaled_array = scale_img(input_array)

        # Check that all pixel values in the scaled array are between 0 and 1
        assert np.all(scaled_array >= 0) and np.all(scaled_array <= 1)

    # Tests that the function scales an input array with all pixel values equal to 0 correctly
    def test_happy_path_all_pixel_values_equal_to_0(self):
        # Create an input array with all pixel values equal to 0
        input_array = np.zeros((3, 3, 3))

        # Call the function under test
        scaled_array = scale_img(input_array)

        # Check that all pixel values in the scaled array are also equal to 0
        assert np.all(scaled_array == 0)

    # Tests that the function scales an input array with all pixel values equal to 1 correctly
    def test_happy_path_all_pixel_values_equal_to_1(self):
        # Create an input array with all pixel values equal to 1
        input_array = np.ones((3, 3, 3))

        # Call the function under test
        scaled_array = scale_img(input_array)
        print(scaled_array)
        # Check that all pixel values in the scaled array are also equal to 0
        assert np.all(scaled_array == 0)

    # Tests that the function handles an input array with negative pixel values correctly
    def test_edge_case_negative_pixel_values(self):
        # Create an input array with negative pixel values
        input_array = np.array([[[-0.2, -0.4, -0.6], [-0.8, -0.1, -0.9]]])

        # Call the function under test
        scaled_array = scale_img(input_array)

        # Check that all pixel values in the scaled array are between 0 and 1
        assert np.all(scaled_array >= 0) and np.all(scaled_array <= 1)

    # Tests that the function handles an input array with pixel values greater than 1 correctly
    def test_edge_case_pixel_values_greater_than_1(self):
        # Create an input array with pixel values greater than 1
        input_array = np.array([[[1.2, 1.4, 1.6], [1.8, 2.0, 2.2]]])

        # Call the function under test
        scaled_array = scale_img(input_array)

        # Check that all pixel values in the scaled array are between 0 and 1
        assert np.all(scaled_array >= 0) and np.all(scaled_array <= 1)

    # Tests that the function scales an input array with non-uniform pixel values correctly
    def test_other_input_array_with_non_uniform_pixel_values(self):
        # Create an input array with non-uniform pixel values
        input_array = np.array([[[0.2, 0.4, 0.6], [0.8, 1.0, 2.0]]])

        # Call the function under test
        scaled_array = scale_img(input_array)

        # Check that all pixel values in the scaled array are between 0 and 1
        assert np.all(scaled_array >= 0) and np.all(scaled_array <= 1)


class TestPreprocessReverse(unittest.TestCase):

    # Tests that the function works correctly with a valid numpy.ndarray image array with shape(batch, height, width, 3)
    def test_valid_image_array(self):
        # Create a valid image array with shape (batch, height, width, 3)
        img = np.zeros((1, 100, 100, 3))
        # Call the preprocess_reverse function
        result = preprocess_reverse(img)
        # Assert that the result has the same shape as the input image
        assert result.shape == img.shape
        # Assert that the result is not equal to the input image
        assert not np.array_equal(result, img)

    # Tests that the function works correctly with a valid numpy.ndarray image array with different height and width
    def test_different_dimensions(self):
        # Create a valid image array with different height and width values
        img = np.zeros((1, 200, 150, 3))
        # Call the preprocess_reverse function
        result = preprocess_reverse(img)
        # Assert that the result has the same shape as the input image
        assert result.shape == img.shape
        # Assert that the result is not equal to the input image
        assert not np.array_equal(result, img)

    # Tests that the function works correctly with a valid numpy.ndarray image array with different channel values
    def test_different_channels(self):
        # Create a valid image array with different channel values
        img = np.zeros((1, 100, 100, 4))
        # Call the preprocess_reverse function
        result = preprocess_reverse(img)
        # Assert that the result has the same shape as the input image
        assert result.shape == img.shape
        # Assert that the result is not equal to the input image
        assert not np.array_equal(result, img)

    # Tests that the function handles an empty numpy.ndarray image array correctly
    def test_empty_image_array(self):
        # Create an empty image array
        img = np.empty((1, 0, 0, 3))
        # Call the preprocess_reverse function
        try:
            result = preprocess_reverse(img)
        except ValueError as e:
            assert str(e) == "Input image array is empty"
        else:
            raise AssertionError("Expected ValueError not raised")

    # Tests that the function handles a numpy.ndarray image array with an invalid shape correctly
    def test_invalid_shape(self):
        # Create an image array with an invalid shape
        img = np.zeros((100, 100))
        # Call the preprocess_reverse function and expect a ValueError to be raised
        with pytest.raises(ValueError):
            preprocess_reverse(img)


if __name__ == '__main__':
    unittest.main()