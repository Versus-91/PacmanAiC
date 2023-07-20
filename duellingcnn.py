import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingConv2dNetwork(nn.Module):
    def __init__(self):
        super(DuelingConv2dNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc_v = nn.Linear(32 * 31 * 28, 256)  # Value stream fully connected layer
        self.relu3_v = nn.ReLU()
        self.fc_a = nn.Linear(32 * 31 * 28, 256)  # Advantage stream fully connected layer
        self.relu3_a = nn.ReLU()
        self.fc_value = nn.Linear(256, 1)  # Final value layer
        self.fc_advantage = nn.Linear(256, 4)  # Final advantage layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)

        v = self.fc_v(x)
        v = self.relu3_v(v)
        v = self.fc_value(v)

        a = self.fc_a(x)
        a = self.relu3_a(a)
        a = self.fc_advantage(a)

        # Combine value and advantage streams
        q = v + (a - torch.mean(a, dim=1, keepdim=True))
        return q