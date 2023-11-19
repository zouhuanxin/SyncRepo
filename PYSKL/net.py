import torch
import torch.nn as nn
import torchvision.models as models

from model.TCN import *

"""
利用2d热力图来表达空间的关系
利用时间轨迹图来表达时间的关系或者这里直接导入骨架点时间流
最后模型应该是一个双流
"""


class PysklModel(nn.Module):
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.resnet18 = models.resnet18()
        modules = list(self.resnet18.children())[:-3]
        self.resnet18 = nn.Sequential(*modules)
        self.pool1 = nn.AvgPool2d(kernel_size=3)
        self.fc1 = nn.Linear(26 * 256 * 7 * 7, 60)
        self.relu1 = nn.ReLU()

        self.tcn1 = unit_tcn(3, 3, kernel_size=11, stride=1)
        self.tcn2 = unit_tcn(3, 3, kernel_size=9, stride=1)
        self.tcn3 = unit_tcn(3, 3, kernel_size=7, stride=1)
        self.tcn4 = unit_tcn(3, 3, kernel_size=5, stride=1)
        self.fc2 = nn.Linear(3 * 300 * 25, 60)
        self.relu2 = nn.ReLU()

    def forward(self, x, skeleton_x):
        spial_out = self.resnet18(x)
        spial_out = spial_out.reshape(self.batch_size, -1)
        spial_out = self.fc1(spial_out)
        spial_out = self.relu1(spial_out)

        temporal_out = self.tcn1(skeleton_x)
        temporal_out = self.tcn2(temporal_out)
        temporal_out = self.tcn3(temporal_out)
        temporal_out = self.tcn4(temporal_out)
        temporal_out = temporal_out.reshape(self.batch_size, -1)
        temporal_out = self.fc2(temporal_out)
        temporal_out = self.relu2(temporal_out)

        out = spial_out + temporal_out

        return out
