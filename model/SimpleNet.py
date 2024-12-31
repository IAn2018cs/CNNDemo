import os
import sys

import torch
import torch.nn as nn

# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.realpath(__file__))
# 将父目录添加到模块搜索路径中
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from my_tool.tools import forward, print_net_shape


class EmptyNet(nn.Module):
    def forward(self, x):
        return x


class SimpleBlock(nn.Module):
    def __init__(self, num_cov, input_channels, output_channels, stride):
        super().__init__()
        blks = []
        for i in range(num_cov - 1):
            blks.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
            input_channels = output_channels
        blks.append(nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, stride=stride))
        if input_channels != output_channels:
            self.transfer_conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=stride)
        else:
            self.transfer_conv = EmptyNet()

        self.net = nn.Sequential(*blks)

    def forward(self, x):
        for net in self.net:
            x = forward(net, x)
        x_hat = self.transfer_conv(x)
        return x_hat + x


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=5, padding=5), nn.BatchNorm2d(64), nn.ReLU(),
            SimpleBlock(2, 64, 128, 2),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 64), nn.Dropout(0.5),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 16), nn.Dropout(0.5),
            nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 2), nn.Dropout(0.5),
        )

    def forward(self, x):
        for net in self.net:
            x = forward(net, x)
        return x


net_name = "SimpleNet"

if __name__ == '__main__':
    x = torch.rand((64, 3, 224, 224))
    simple_net = SimpleNet()
    print_net_shape(simple_net.net, x)
