import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.cnn import CNNModel
import os



def train_model():
    print("DEBUG: train_model() function called.")
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.0001
    num_epochs = 100  # Increased to allow more training time
    patience = 10
    min_delta = 0.001
    model_save_path = 'models/cnn_model.pth'

    # Data transformations
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # Random rotation
        transforms.RandomAffine(0, translate=(0.2, 0.2)),  # Random translation
        transforms.RandomAffine(0, scale=(0.9, 1.1)),  # Random scaling
        transforms.RandomAffine(0, shear=10),  # Random shear
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Load datasets
    train_dataset = datasets.MNIST(root='data/processed', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Use more available CPU cores
    )

    # Initialize model, loss function, and optimizer
    model = CNNModel()  # No need to move to GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    best_loss = float('inf')
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)  # Average loss for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

        # Check for improvement
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with loss: {best_loss:.6f}")
        else:
            epochs_without_improvement += 1

        # Break if no improvement for 'patience' epochs
        if epochs_without_improvement >= patience:
            print("Early stopping triggered. Training stopped.")
            break

    print(f"Training completed. Best loss: {best_loss:.4f}")




if __name__ == "__main__":

    os.makedirs("models", exist_ok=True)
    train_model()

