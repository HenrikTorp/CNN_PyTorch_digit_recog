# Digit Recognizer with PyTorch

This project implements a **Convolutional Neural Network (CNN)** using PyTorch to recognize handwritten digits. The model processes images more effectively by leveraging convolutional layers, which focus on local patterns in the image (e.g., edges and shapes).

It achieves much better results than the fully connected network I previously made using only NumPy, reaching 99.21% accuracy on a dataset of custom drawn digits after being trained on the MNIST dataset. This shows the model has generalized well and performs strongly on new data even without any fine-tuning.

Another significant contributor in this improvement was data augmentation: by applying random distortions to the digits in the MNIST dataset, the effective size of the training set was increased and the model became more robust to variations. When drawing digits, the CNN is noticeably better at handling distorted or unusual digits than the previous fully connected neural network, which did not use data augmentation.

## Features
- **Two Convolutional Layers:** The first layer processes the input image in 3x3 chunks, extracting local features. The second layer refines these features before passing them to fully connected layers.
- **ReLU Activation:** ReLU is used instead of sigmoid. f(x) = max(0, x). It outputs the input directly if it's positive, and 0 otherwise. Computationally simpler than sigmoid, and it is less prone to vanishing gradients.
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

## How to run

1. **Install requirements:**
   ```
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```
   python src/train.py
   ```

3. **(Optional) Evaluate the model:**
   ```
   python model_eval.py
   ```

4. **Run the digit drawing GUI:**
   ```
   python main.py
   ```

Make sure you have Python 3.8+ installed.  
The GUI will let you draw digits and see the model’s predictions in real time.


