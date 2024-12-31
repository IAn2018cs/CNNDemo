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


def conv_blk(input_channels, output_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
    )


class DenseBlock(nn.Module):
    def __init__(self, num_conv, input_channels, output_channels):
        super().__init__()
        layer = []
        for i in range(num_conv):
            layer.append(conv_blk(i * output_channels + input_channels, output_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            x = torch.cat((x, y), dim=1)
        return x


class TransitionBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        # 1x1卷积改变通道数，池化层改变宽度
        self.net = nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        for net in self.net:
            x = net(x)
        return x


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        )
        input_channels, growth_rate = 64, 32
        num_conv_in_dense_blocks = [2, 2]
        blks = []
        for i, num_conv in enumerate(num_conv_in_dense_blocks):
            blks.append(DenseBlock(num_conv, input_channels, growth_rate))
            # 上一个稠密块的输出通道数
            input_channels += num_conv * growth_rate
            # 在稠密块之间加一个转换层，让通道数减半
            if i != len(num_conv_in_dense_blocks) - 1:
                blks.append(TransitionBlock(input_channels, input_channels // 2))
                input_channels = input_channels // 2
        b2 = nn.Sequential(*blks)
        b3 = nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.net = nn.Sequential(b1, b2, b3, nn.BatchNorm1d(128),
                                 nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0),
                                 nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0),
                                 nn.Linear(32, 2))

    def forward(self, x):
        for net in self.net:
            x = forward(net, x)
        return x


net_name = "DenseNet"

__all__ = ["DenseNet", "net_name"]

if __name__ == '__main__':
    x = torch.rand((64, 3, 224, 224))
    # dense_blk = DenseBlock(3, 3, 16)
    # transition_blk = TransitionBlock(51, 10)
    # y = dense_blk(x)
    # y_hat = transition_blk(y)
    # print(y.shape)
    # print(y_hat.shape)
    dense_net = DenseNet()
    print_net_shape(dense_net.net, x)
