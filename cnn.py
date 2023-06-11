import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(32 * 20 * 28, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        batch_size = x.size(0)
        # Concatenate along the channel dimension
        x = x.view(batch_size, 6 * 4, 22, 18)
        # Define the forward pass of the neural network
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(32 * 22 * 18, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
