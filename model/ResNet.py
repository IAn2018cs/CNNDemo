import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.realpath(__file__))
# 将父目录添加到模块搜索路径中
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from my_tool.tools import forward, print_net_shape


class Residual(nn.Module):
    def __init__(self, input_channels, output_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channel,
                               kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel,
                               kernel_size=3, padding=1)

        if input_channels != output_channel or stride != 1:
            self.conv3 = nn.Conv2d(input_channels, output_channel, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        x_hat = F.relu(self.bn1(self.conv1(x)))
        x_hat = self.bn2(self.conv2(x_hat))
        if self.conv3:
            y = x_hat + self.conv3(x)
        else:
            y = x_hat + x
        return F.relu(y)


def residual_blk(input_channels, output_channels, num_res, stride):
    blk =[]
    for i in range(num_res):
        if i == 0:
            stride = stride
        else:
            stride = 1
        blk.append(Residual(input_channels, output_channels, stride))
        input_channels = output_channels
    return blk


b1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1), nn.Dropout(0.5),
                       )
b2 = nn.Sequential(*residual_blk(64, 128, 2, stride=2))
b3 = nn.Sequential(*residual_blk(128, 256, 2, stride=2))
b4 = nn.Sequential(*residual_blk(256, 512, 2, stride=2))


resnet = nn.Sequential(
    b1, b2, b3, b4, nn.MaxPool2d(kernel_size=3, stride=2),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 2)
)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet

    def forward(self, x):
        for net in self.net:
            x = forward(net, x)
        return x


net_name = "ResNet"

__all__ = ["ResNet", "net_name"]

if __name__ == '__main__':
    x = torch.rand((32, 3, 224, 224))
    print_net_shape(resnet, x)

