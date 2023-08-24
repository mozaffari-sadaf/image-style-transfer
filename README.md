# Image Style Transfer

Apply artistic styles to images using machine learning techniques. This project demonstrates how to transfer the style of one image onto the content of another, creating stunning artistic effects.

## Introduction

Image Style Transfer is a machine learning project that leverages deep learning and neural networks to apply artistic styles to images. It uses pre-trained models to capture content and style features from input images and generate unique stylized outputs.

## Features

- Transfer artistic styles to images.
- Easy setup and usage.
- Customizable settings for experimenting with different styles.
- Python code using TensorFlow and Keras.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/mozaffari-sadaf/image-style-transfer.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
## Usage
1. Run the style transfer script:
   ```
   python transfer_style.py --content content.png --style style.png
   ```
   Replace content.jpg with the path to your content image and style.jpg with the path to your style image.
2. The script will generate a stylized image that combines the content of the content image with the style of the style image.

## Examples
### Sample Content and Style Images

In this project, I've provided sample content images and style images to help you get started. You can find them in the `images` directory.

### Usage Example

To apply a style to a content image using one of the provided samples, you can run the following command:

```
python transfer_style.py --content city.png --style style1.png
```
Here is what you will get:
![Style Transfer Example](https://github.com/mozaffari-sadaf/image-style-transfer/assets/49075210/1b396342-4071-4828-b686-605ced68d68f)
## Contributing
Contributions are welcome! Feel free to open pull requests or report issues.

