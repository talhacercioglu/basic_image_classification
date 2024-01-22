# CIFAR-10 Image Classification

This repository contains a Python script for basic image classification on the CIFAR-10 dataset using TensorFlow and Keras. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The script builds a Convolutional Neural Network (CNN) for image classification and provides insights into its training performance.

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo](https://github.com/talhacercioglu/basic_image_classification.git

2. **Install Dependencies:**
   Ensure that you have the required dependencies installed. You can install them using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Preparation:**
   The CIFAR-10 dataset is automatically loaded using TensorFlow's dataset utilities. No additional steps are required for dataset preparation.

4. **Run the Script:**
   Execute the main script to train the model and visualize its performance:
   ```bash
   python cifar10_classification.py
   ```

## Model Architecture

The Convolutional Neural Network (CNN) architecture used for image classification consists of multiple convolutional layers, batch normalization, max pooling, dropout, and fully connected layers. The model is compiled using the Adam optimizer and categorical crossentropy as the loss function. Early stopping is implemented to prevent overfitting.

## Training and Evaluation

The model is trained for 50 epochs with data augmentation using an ImageDataGenerator. Training and validation metrics such as accuracy, precision, and recall are monitored and visualized to evaluate the model's performance.

## Results Visualization

The training progress is visualized using Matplotlib subplots, showcasing the evolution of the loss, accuracy, precision, and recall over the training epochs.

## Notes

- The script includes commented-out sections for potential enhancements, such as early stopping and adjusting batch size.
- Most of these codes were taken from Kaggle and rearranged with my individual comments!

Feel free to experiment with the code and parameters to further optimize the model's performance on the CIFAR-10 dataset.
