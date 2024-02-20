# PNEUMONIA-PREDICTION
Binary classification of Chest X-ray images (Pneumonia/Normal) using a custom CNN, transfer learning with pre-trained models, and fine-tuning over 150+ hours. Includes a Gradio demo for interactive model exploration.
## Overview

This repository contains a machine learning project focused on predicting pneumonia in Chest X-ray images. The project utilizes a custom Convolutional Neural Network (CNN) as well as transfer learning with pre-trained models. The models have been fine-tuned to achieve optimal performance.

## Features

- Custom CNN for Chest X-ray image classification
- Transfer learning using pre-trained models
- Fine-tuning of models for improved accuracy
- Gradio demo for interactive exploration of model predictions

## Getting Started

### Note on Interactive Features

Please be aware that certain interactive features, such as the Gradio demo and clickable index, may not function as intended when viewing this project directly on GitHub. Gradio APIs may require a live Python kernel to function properly.

### Note on Collapsed Cells and Screenshots

To enhance readability, several cells in the Jupyter notebooks have been collapsed for a more concise and organized view. However, when viewing the notebooks on GitHub, all cells may appear expanded.

Additionally, for a visual representation of the Gradio demo in action, a screenshot has been provided in the repository. You can find it in the [Repository](Gradio.png).

For the most accurate representation of the interactive elements and the Gradio demo, it is recommended to run the Jupyter notebooks locally in a Jupyter environment.

### Prerequisites

- Python 3.x
- TensorFlow
- Gradio
- Other dependencies (requirements.txt)

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Kishor-Gulati/PNEUMONIA-PREDICTION.git
    cd PNEUMONIA-PREDICTION
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Gradio demo:**

    ```bash
    jupyter nbconvert --to notebook --execute --inplace gradio.ipynb && jupyter-notebook gradio.ipynb
    ```

### Model Training

#### Dataset Preparation

**Dataset Source:**
The Chest X-ray dataset used for training is obtained from [Mendeley Chest X-ray Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/3).

**Data Splitting:**
The dataset is split into training, validation, and test sets for model evaluation.

#### Model Architecture

**Custom CNN:**
A custom Convolutional Neural Network (CNN) is designed for Chest X-ray image classification. The architecture includes several convolutional layers, pooling layers, and dense layers. For specific details, refer to the model architecture section in the code.

**Transfer Learning:**
Transfer learning is applied using pre-trained models. The selected pre-trained models are ResNet152V2, DenseNet201, InceptionV3, MobileNetV2, and VGG16. These models provide a strong foundation for feature extraction.

**Fine-Tuning:**
Fine-tuning is performed to optimize the model's performance for pneumonia prediction. The final layers of the pre-trained models are fine-tuned to adapt them to the specific characteristics of the Chest X-ray dataset.

### Training Process

**Data Augmentation:**
Data augmentation techniques such as rotation, flipping, and zooming are applied to increase dataset variability. This helps improve the model's ability to generalize to different variations of Chest X-ray images.

**Optimizer and Loss Function:**
The Adam optimizer is used for training, and the binary_crossentropy loss function is employed for binary classification tasks like pneumonia prediction.

**Learning Rate Adjustment:**
The learning rate is adjusted during training using the ReduceLROnPlateau technique. This dynamic adjustment helps in reaching the optimal minima and speeds up convergence.

**Callbacks:**
Callbacks, including ReduceLROnPlateau and EarlyStopping, are employed during training. ReduceLROnPlateau adjusts the learning rate based on the model's performance, and EarlyStopping halts training when the model stops improving.

**Training Duration:**
The training duration spans over 150+ hours. The long training duration indicates the effort taken to ensure the model reaches optimal performance.

## Usage

To seamlessly integrate the trained models into your applications, follow these simplified steps:

### Loading the Model

Use the following code to load the trained model into your application:

```python
from tensorflow.keras.models import load_model

# Load the model
model = load_model("path/to/your/trained/model.h5")
```

## Acknowledgments

This project wouldn't have been possible without the valuable contributions and resources from the following:

- [Kaggle](https://www.kaggle.com/): Providing a platform for datasets and collaborative data science.
- [Imarticus Learning](https://imarticus.org/): The institute that contributed to the learning and development of the project.
- [TensorFlow](https://www.tensorflow.org/): The powerful open-source machine learning framework.
- [Keras](https://keras.io/): A high-level neural networks API used as a frontend for TensorFlow.
- [Gradio](https://www.gradio.app/): Simplifying the deployment of machine learning models with interactive interfaces.
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/guide/keras/transfer_learning): A comprehensive guide on transfer learning with TensorFlow.
- [ImageDataGenerator Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator): Information about image data augmentation in Keras.
- [Keras Applications](https://keras.io/api/applications/): Documentation on Keras pre-trained models and applications.
- [ResNet152V2 Documentation](https://keras.io/api/applications/resnet/#resnet152v2-function): Details about the ResNet152V2 pre-trained model in Keras.
- [VGG Documentation](https://keras.io/api/applications/vgg/): Documentation on the VGG pre-trained model in Keras.
- [MobileNet Documentation](https://keras.io/api/applications/mobilenet/): Documentation on the MobileNet pre-trained model in Keras.
- [InceptionV3 Documentation](https://keras.io/api/applications/inceptionv3/): Documentation on the InceptionV3 pre-trained model in Keras.
- [DenseNet Documentation](https://keras.io/api/applications/densenet/): Documentation on the DenseNet pre-trained model in Keras.
- [LIME (Local Interpretable Model-agnostic Explanations)](https://github.com/marcotcr/lime?tab=readme-ov-file): A library for explaining machine learning models.
- [YouTube Video - Transfer Learning with TensorFlow](https://www.youtube.com/watch?v=hUnRCxnydCc): A tutorial video on transfer learning with TensorFlow.
- [LIME Documentation](https://lime-ml.readthedocs.io/en/latest/): Official documentation for LIME.

These resources have been instrumental in the development and success of the project. Feel free to explore them for further insights and learning.

## Contact

For any inquiries, contact [Kishor Gulari] at [kanushgulati@gmail.com].
