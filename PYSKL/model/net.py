import torch
import torch.nn as nn
import torchvision.models as models

from .TCN import *

"""
利用2d热力图来表达空间的关系
利用时间轨迹图来表达时间的关系或者这里直接导入骨架点时间流
最后模型应该是一个双流
"""


class PysklModel(nn.Module):
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=9, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=1)
        self.norm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=7, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=1)
        self.norm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=5, stride=1)
        self.norm4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.fc1 = nn.Linear(26 * 512 * 13 * 13, 60)
        self.spial_relu = nn.ReLU()

        self.tcn1 = unit_tcn(3, 3, kernel_size=11, stride=1)
        self.tcn2 = unit_tcn(3, 3, kernel_size=9, stride=1)
        self.tcn3 = unit_tcn(3, 3, kernel_size=7, stride=1)
        self.tcn4 = unit_tcn(3, 3, kernel_size=5, stride=1)
        self.fc2 = nn.Linear(3 * 300 * 25, 60)
        self.temporal_relu = nn.ReLU()

    def forward(self, x, skeleton_x):
        spial_out = self.relu1(self.norm1(self.pool1(self.conv1(x))))
        spial_out = self.relu2(self.norm2(self.pool2(self.conv2(spial_out))))
        spial_out = self.relu3(self.norm3(self.pool3(self.conv3(spial_out))))
        spial_out = self.relu4(self.norm4(self.pool4(self.conv4(spial_out))))
        spial_out = spial_out.reshape(self.batch_size, -1)
        spial_out = self.fc1(spial_out)
        spial_out = self.spial_relu(spial_out)

        temporal_out = self.tcn1(skeleton_x)
        temporal_out = self.tcn2(temporal_out)
        temporal_out = self.tcn3(temporal_out)
        temporal_out = self.tcn4(temporal_out)
        temporal_out = temporal_out.reshape(self.batch_size, -1)
        temporal_out = self.fc2(temporal_out)
        temporal_out = self.temporal_relu(temporal_out)

        out = spial_out + temporal_out

        return out

