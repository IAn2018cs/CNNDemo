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


def nin_block(input_chanel, output_chanel, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(input_chanel, output_chanel, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(output_chanel), nn.ReLU(),
        nn.Conv2d(in_channels=output_chanel, out_channels=output_chanel, kernel_size=1), nn.BatchNorm2d(output_chanel),
        nn.ReLU(),
        nn.Conv2d(in_channels=output_chanel, out_channels=output_chanel, kernel_size=1), nn.BatchNorm2d(output_chanel),
        nn.ReLU()
    )


nin = nn.Sequential(
    nin_block(3, 96, kernel_size=7, stride=2, padding=3),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(384, 2, kernel_size=3, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

net_name = "NiN"


class NinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nin

    def forward(self, x):
        for net in self.net:
            x = forward(net, x)
        return x


__all__ = ["NinNet", "net_name"]

if __name__ == '__main__':
    x = torch.rand((64, 3, 64, 64))
    print_net_shape(nin, x)
