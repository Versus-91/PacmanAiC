import torch
import torch.nn as nn



class Conv2dNetwork(nn.Module):
    def __init__(self):
        super(Conv2dNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(32 * 31 * 28, 128)
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
class PacmanNet(nn.Module):
    def __init__(self):
        super(PacmanNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc_v = nn.Linear(32 * 31 * 28, 256)
        self.relu_v = nn.ReLU()
        self.fc_a = nn.Linear(32 * 31 * 28, 256)
        self.relu_a = nn.ReLU()
        self.fc_value = nn.Linear(256, 1)
        self.fc_advantage = nn.Linear(256, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)

        v = self.fc_v(x)
        v = self.relu_v(v)
        v = self.fc_value(v)

        a = self.fc_a(x)
        a = self.relu_a(a)
        a = self.fc_advantage(a)


        q = v + (a - torch.mean(a, dim=1, keepdim=True))

        return q