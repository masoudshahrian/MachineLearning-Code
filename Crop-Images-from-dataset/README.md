# Face Crop from Dataset

This project automatically detects and crops faces from the CelebA dataset (or any image dataset) using dlib face detection and OpenCV. The cropped faces are centered and resized to 128x128 pixels.

## Features

- **Automatic Face Detection**: Uses dlib's frontal face detector to identify faces in images
- **Center-based Cropping**: Crops a 128x128 pixel region centered on detected faces
- **Batch Processing**: Processes entire directories of images
- **Support for Multiple Formats**: Works with both JPG and PNG images

## How It Works

1. Loads images from the input directory
2. Converts each image to grayscale for face detection
3. Detects faces using dlib's face detector
4. Calculates the center of each detected face
5. Crops a 128x128 pixel region around the face center
6. Saves the cropped and resized image to the output directory

## Requirements

- Python 3.6+
- OpenCV (cv2)
- dlib
- NumPy (dependency of OpenCV)

## Installation

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. **Configure Input/Output Paths**: 
   Edit the `input_dir` and `output_dir` variables in the notebook to point to your dataset location:
   ```python
   input_dir = '/path/to/your/images/'
   output_dir = '/path/to/save/cropped/images/'
   ```

2. **Run the Notebook**:
   Open `Crop_Image_from_Dataset.ipynb` in Jupyter Notebook or Google Colab and run all cells.

3. **Check Results**:
   Cropped images will be saved in the output directory with the same filenames as the original images.

## Google Colab

This notebook is designed to work with Google Colab and Google Drive. Click the badge below to open in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masoudshahrian/MachineLearning-Code/blob/master/Crop_Image_from_Dataset.ipynb)

## Technical Details

- **Crop Size**: 128x128 pixels (centered on face)
- **Face Detection**: dlib's HOG-based frontal face detector
- **Image Processing**: OpenCV (cv2)
- **Supported Formats**: JPG, PNG

## Notes

- The script processes one face per image. If multiple faces are detected, all will be processed but the output will be overwritten with the last detected face.
- The cropping algorithm ensures that the cropped region stays within image boundaries.
- Original aspect ratio is not preserved due to the fixed 128x128 output size.

## License

This code is provided as-is for educational and research purposes.

## Original Dataset

This project is designed to work with the CelebA dataset, but can be adapted for any image dataset with faces.

