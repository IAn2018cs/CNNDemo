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

alex_net = nn.Sequential(
    nn.Conv2d(3, 48, 7, 2, 3), nn.BatchNorm2d(48), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(48, 96, padding=2, kernel_size=5), nn.BatchNorm2d(96), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
    nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
    nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(864, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(64, 2)
)

net_name = "AlexNet"


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = alex_net

    def forward(self, x):
        for net in self.net:
            x = forward(net, x)
        return x


__all__ = ["AlexNet", "net_name"]

if __name__ == '__main__':
    x = torch.rand((64, 3, 64, 64))
    print_net_shape(alex_net, x)
