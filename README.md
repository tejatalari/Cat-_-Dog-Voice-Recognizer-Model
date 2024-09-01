

# Cat-Dog Audio Classification Using Convolutional Neural Network (CNN)

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project focuses on classifying audio clips as either cat or dog sounds using a Convolutional Neural Network (CNN). The primary goal is to develop a model that can accurately distinguish between these two types of animal sounds.

## Dataset

The dataset consists of audio clips categorized into two classes:
1. Cat Sounds
2. Dog Sounds

### Dataset Preparation

The audio files are processed to extract features like spectrograms, which are then used as input to the CNN model.

### Dataset Structure

- **Training Data**: Contains audio files for training the model.
- **Testing Data**: Contains audio files for evaluating the model's performance.

## Model Architecture

The model is built using a Convolutional Neural Network (CNN) to classify audio files based on their spectrogram representations.

### Key Components

- **Input Layer**: Accepts spectrograms extracted from the audio files.
- **Convolutional Layers**: Extract features from the input spectrograms.
- **Pooling Layers**: Reduce the spatial dimensions of the feature maps.
- **Dense Layers**: Classify the extracted features into cat or dog sounds.
- **Softmax Activation**: Used in the final layer to output probabilities for each class.

### Model Summary

- **Conv2D**: 32 filters, kernel size 3x3, ReLU activation
- **MaxPooling2D**: Pool size 2x2
- **Conv2D**: 64 filters, kernel size 3x3, ReLU activation
- **MaxPooling2D**: Pool size 2x2
- **Flatten Layer**
- **Dense Layer**: 128 units, ReLU activation
- **Output Layer**: 2 units, Softmax activation

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/cat-dog-audio-classify-conv2d.git
    ```
2. Navigate to the project directory:
    ```bash
    cd cat-dog-audio-classify-conv2d
    ```
3. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Extract Features**:
    ```bash
    python extract_features.py --input_dir path/to/audio/files --output_dir path/to/save/spectrograms
    ```
2. **Train the Model**:
    ```bash
    python train.py
    ```
3. **Evaluate the Model**:
    ```bash
    python evaluate.py
    ```
4. **Predict on New Audio**:
    ```bash
    python predict.py --audio_file path/to/audio/file.wav
    ```

## Results

The model achieved the following performance metrics on the test dataset:

- **Accuracy**: 88%
- **Precision**: 87%
- **Recall**: 86%
- **F1-Score**: 86%

### Sample Predictions

- Audio 1: Cat (Prediction: Cat, Confidence: 90%)
- Audio 2: Dog (Prediction: Dog, Confidence: 92%)

## Future Work

- **Model Improvement**: Explore different CNN architectures and hyperparameters to improve classification accuracy.
- **Data Augmentation**: Implement audio augmentation techniques to enhance model robustness.
- **Deployment**: Deploy the model as a web service for real-time audio classification.

## Contributing

Contributions are welcome! If you’d like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


 
