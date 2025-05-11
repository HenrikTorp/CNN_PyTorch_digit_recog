# Digit Recognizer with PyTorch

This project implements a **Convolutional Neural Network (CNN)** using PyTorch to recognize handwritten digits. The model is designed to process images more effectively by leveraging convolutional layers, which focus on local patterns in the image (e.g., edges, shapes).

## Features
- **Two Convolutional Layers:** The first layer processes the input image in 3x3 chunks, extracting local features. The second layer refines these features before passing them to fully connected layers.
- **ReLU Activation:** ReLU (Rectified Linear Unit) is used instead of sigmoid for faster convergence and to avoid vanishing gradients.
- **Adam Optimizer:** The Adam optimization algorithm is used instead of vanilla gradient descent for better performance and faster training.
- **Data Augmentation:** Random rotations, translations, and scaling are applied to improve generalization.

## How It Works
1. **Convolutional Layers:**
   - The first convolutional layer extracts features from the input image by dividing it into 3x3 chunks.
   - The second convolutional layer processes the output of the first layer to refine the features.
2. **Fully Connected Layers:**
   - After the convolutional layers, the features are flattened and passed through a fully connected layer to classify the digit.
3. **Training:**
   - The model is trained on the MNIST dataset using the Adam optimizer and a cross-entropy loss function.
   - 