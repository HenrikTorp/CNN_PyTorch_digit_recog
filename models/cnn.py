import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        # improves computational efficiency by halving the spatial dimensions of the input
        # in each 2x2 region, it takes the maximum value, so therfore it becomes easier to calculate, 14 by 14 instead of 28 by 28
        self.dropout = nn.Dropout(0.25) 
        # randomly zeroes some of the elements of the input tensor with probability 0.25
        # prevents overfitting
        # by reducing the model's reliance on any one feature
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # reduces tensor to a batch of vectors 
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    # Adam optimizer being a variation of stochastic gradient descent (SGD)
    # It features momentum and adaptive learning rates, making it efficient for training deep neural networks.