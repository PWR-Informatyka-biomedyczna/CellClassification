import torch
from torch import nn


class CNN(nn.Module):

    def __init__(self, input_channels, num_classes):
        super().__init__()
        # convolution layers
        self.conv1 = nn.Conv2d(input_channels, 16, 3)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 16, 3)

        # fully connected part
        self.flatten = nn.Flatten()
        self.l0 = nn.Linear(12544, 64)
        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, num_classes)

        # activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def conv_pass(self, conv, x):
        return self.relu(conv(x))

    def forward(self, x):
        # first layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mp1(x)
        # second layer
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp2(x)
        # third layer
        x = self.conv3(x)
        x = self.relu(x)
        # dense layers
        x = self.flatten(x)
        x = self.relu(x)
        x = self.l0(x)
        x = self.relu(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        out = self.softmax(x)
        return out
