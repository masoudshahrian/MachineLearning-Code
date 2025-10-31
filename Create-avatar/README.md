# Create Avatar From Image

A Python project that transforms real face images into cartoon/anime-style avatars using deep learning. This project combines MediaPipe for face detection and AnimeGAN2 for style transfer to create beautiful cartoon avatars from photographs.

## Features

- **Face Detection**: Automatically detects faces in images using MediaPipe
- **Cartoon/Anime Conversion**: Transforms real photos into anime-style artwork using AnimeGAN2
- **Easy to Use**: Simple interface with just a few lines of code
- **High Quality**: Uses state-of-the-art deep learning models for high-quality results
- **Google Colab Support**: Includes Colab badge for easy cloud execution

## Demo

The project processes face images through two main steps:
1. Detects faces in the input image using MediaPipe
2. Applies anime/cartoon style transfer using AnimeGAN2 model

## Requirements

- Python 3.7+
- PyTorch
- MediaPipe
- OpenCV
- NumPy
- Matplotlib

See `requirements.txt` for detailed version information.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/masoudshahrian/MachineLearning-Code.git
cd Create-avatar
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install mediapipe torch torchvision opencv-python matplotlib numpy
```

## Usage

### Basic Usage (Display Only)

```python
import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# Load the model
model = torch.hub.load('bryandlee/animegan2-pytorch', 'generator', pretrained='face_paint_512_v2')
model.eval()

# Setup MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Process image
input_image_path = "path/to/your/image.jpg"
output_image = process_image(input_image_path)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(plt.imread(input_image_path)), plt.title('Real Image')
plt.subplot(122), plt.imshow(output_image), plt.title('Cartoon Avatar')
plt.show()
```

### Advanced Usage (Save Output)

```python
# Process and save image
input_image_path = "path/to/your/image.jpg"
output_image_path = "path/to/save/avatar.jpg"
output_image = process_image(input_image_path, output_image_path)
```

### Run in Google Colab

Click the badge below to open and run the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masoudshahrian/MachineLearning-Code/blob/master/Create_Avatar_From_Image.ipynb)

## How It Works

### 1. Face Detection
The project uses MediaPipe's face detection solution to identify faces in the input image. MediaPipe provides fast and accurate face detection with a minimum confidence threshold of 0.5.

### 2. Style Transfer
Once a face is detected, the image is processed through AnimeGAN2 (specifically the `face_paint_512_v2` variant), which applies anime/cartoon style transfer:

- **Preprocessing**: The image is converted to a tensor and normalized
- **Model Inference**: The AnimeGAN2 model generates the cartoon version
- **Postprocessing**: The output is denormalized and converted back to an image format

### 3. Output
The final cartoon avatar is displayed side-by-side with the original image and can optionally be saved to disk.

## Model Details

This project uses:
- **AnimeGAN2**: A GAN-based model specifically trained for transforming photos into anime/cartoon style
- **Pretrained Variant**: `face_paint_512_v2` - optimized for face images at 512x512 resolution
- **MediaPipe Face Detection**: Google's lightweight face detection solution

## Error Handling

If no face is detected in the image, the program will raise an error:
```
ValueError: No face detected in the image!
```

Make sure your input image:
- Contains at least one clearly visible face
- Has good lighting
- Is not too small or low resolution

## Examples

### Input
A regular photograph of a person's face

### Output
An anime/cartoon style avatar with the same facial features and expression

## Customization

You can customize the behavior by:
- Adjusting the `min_detection_confidence` parameter for face detection sensitivity
- Using different AnimeGAN2 variants (e.g., 'celeba_distill', 'face_paint_512_v1')
- Modifying the image preprocessing pipeline

## Limitations

- Works best with clear, front-facing photos
- Requires at least one face in the image
- Processing time depends on image size and hardware
- GPU recommended for faster processing

## Credits

- **AnimeGAN2**: [bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)
- **MediaPipe**: [Google MediaPipe](https://mediapipe.dev/)
- **Original Author**: Masoud Shahrian

## License

This project is open source. Please check the licenses of the underlying models:
- AnimeGAN2: Check the original repository for license details
- MediaPipe: Apache License 2.0

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## Contact

For questions or feedback, please open an issue on the GitHub repository.

## Acknowledgments

Special thanks to:
- The creators of AnimeGAN2 for the amazing style transfer model
- Google for the MediaPipe framework
- The PyTorch team for the deep learning framework

