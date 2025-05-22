from models.cnn import CNNModel
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(data_path, batch_size=32):
    """
    Load data from a folder using PyTorch's ImageFolder.

    Args:
        data_path (str): Path to the folder containing the dataset.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert to tensor
        transforms.Lambda(lambda x: 1.0 - x),  # Invert colors
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Load the dataset using ImageFolder
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the given dataset.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        float: Accuracy of the model on the dataset.
    """
    model.to(device) # Move model to the specified device
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

from PIL import Image
import torch
from torchvision import transforms

def load_single_image(image_path, device):
    """
    Load and preprocess a single image for model evaluation.

    Args:
        image_path (str): Path to the image file.
        device (str): Device to move the image tensor to ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input.
    """
    # Define transformations for the image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert to tensor
        transforms.Lambda(lambda x: 1.0 - x),  # Invert colors
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Open the image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    return image_tensor


"""
image = load_single_image('dutch8test.png', device='cpu')
model = CNNModel()
model.load_state_dict(torch.load("models/cnn_model.pth", map_location='cpu'))
#print predicted label
with torch.no_grad():
    model.eval()
    output = model(image)
    predicted_label = torch.argmax(output, dim=1).item()
    print(f"Predicted label: {predicted_label}")
"""




if __name__ == "__main__":
    # Path to the dataset folder
    data_path = "images/"  # Replace with the full path
    model_path = "models/cnn_model.pth"  # Path to the saved model

    # Load the dataset
    dataloader = load_data(data_path, batch_size=32)

    # Load the trained model
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))

    # Evaluate the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    accuracy = evaluate_model(model, dataloader, device)
    print(f"Accuracy on the dataset: {accuracy * 100:.2f}%")



