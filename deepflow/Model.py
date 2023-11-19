from utils_zhx import *
import torch
import torch.nn as nn


class SpatialStreamConvNet(nn.Module):
    def __init__(self, height, width, channels):
        super().__init__()
        nn.LSTMCell
        self.conv1 = nn.Conv2d(channels, 96, kernel_size=7, stride=2)
        self.norm1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        self.norm2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1)

        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(14336, 4096)
        self.drop1 = nn.Dropout()

        self.fc2 = nn.Linear(4096, 101)
        self.drop2 = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.pool2(x)

        x = self.conv3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)

        x = nn.Softmax(x)

        return x.dim


class TemporalStreamConvNet(nn.Module):
    def __init__(self, height, width, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 96, kernel_size=7, stride=2)
        self.norm1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        self.norm2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1)

        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(-1, 4096)
        self.drop1 = nn.Dropout()

        self.fc2 = nn.Linear(4096, 2048)
        self.drop2 = nn.Dropout()

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.pool2(x)

        x = self.conv3(x)

        x = self.conv4(x)
        x = self.norm4(x)

        x = self.fc1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)

        x = self.softmax(x)

        return x.dim


class TwoStream_ActionRecognition(nn.Module):
    def __init__(self, height, width, channels):
        self.height = height
        self.width = width
        self.channels = channels

        self.SpatialStreamConvNet = SpatialStreamConvNet(height, width, channels)
        self.TemporalStreamConvNet = TemporalStreamConvNet(height, width, channels)

    def forward(self, x1, x2):
        outputs1 = self.SpatialStreamConvNet(x1)
        outputs2 = self.TemporalStreamConvNet(x2)

        # fusion
        outputs3 = [outputs1, outputs2]
        print(outputs3)
