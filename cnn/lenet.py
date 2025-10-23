import torch.nn as nn
import torch
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.conv_3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv_3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x