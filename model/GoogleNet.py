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


class Inception(nn.Module):
    def __init__(self, input_channels, c1, c2, c3, c4):
        super().__init__()
        # 线路一, 单层1x1的卷积层
        self.p_1_1 = nn.Conv2d(in_channels=input_channels, out_channels=c1, kernel_size=1)
        self.b_1 = nn.BatchNorm2d(c1)
        # 线路二，单层1x1卷积层接3x3卷积层
        self.p_2_1 = nn.Conv2d(in_channels=input_channels, out_channels=c2[0], kernel_size=1)
        self.p_2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)
        self.b_2 = nn.ModuleList((nn.BatchNorm2d(c2[0]), nn.BatchNorm2d(c2[1])))
        # 线路三，单层1x1卷积层接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels=input_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)
        self.b_3 = nn.ModuleList((nn.BatchNorm2d(c3[0]), nn.BatchNorm2d(c3[1])))
        # 线路四, 单层3x3最大池化层接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels=input_channels, out_channels=c4, kernel_size=1)
        self.b_4 = nn.BatchNorm2d(c4)

    def forward(self, x):
        p1 = self.b_1(self.p_1_1(x))
        p2 = self.b_2[1](self.p_2_2(F.relu(self.b_2[0](self.p_2_1(x)))))
        p3 = self.b_3[1](self.p3_2(F.relu(self.b_3[0](self.p3_1(x)))))
        p4 = self.b_4(self.p4_2(self.p4_1(x)))

        return torch.cat((p1, p2, p3, p4), dim=1)


google_net = nn.Sequential(
    nn.Conv2d(3, out_channels=64, kernel_size=7, padding=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    Inception(128, 64, (16, 64), (16, 64), 64),
    Inception(256, 128, (64, 128), (64, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
    Inception(128, 32, (16, 32), (16, 32), 32), nn.BatchNorm2d(128), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(in_features=128 * 3 * 3, out_features=128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(128, 2)
)

net_name = "GoogleNet"


class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = google_net

    def forward(self, x):
        for net in self.net:
            x = forward(net, x)
        return x


__all__ = ["GoogleNet", "net_name"]

if __name__ == '__main__':
    x = torch.rand((32, 3, 64, 64))
    print_net_shape(google_net, x)
