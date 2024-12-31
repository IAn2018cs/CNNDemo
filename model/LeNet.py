import os
import sys

from Conv import *

# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.realpath(__file__))
# 将父目录添加到模块搜索路径中
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from my_tool.tools import forward

# net = nn.Sequential(Conv2d(3, 6, 5, 2, 2), nn.ReLU(),
#                     MaxPool2d(2, stride=2),
#                     Conv2d(6, 12, 3, 1, 1), nn.ReLU(),
#                     MaxPool2d(2, stride=2),
#                     Conv2d(12, 12, 3, 1, 1), nn.ReLU(),
#                     MaxPool2d(2, stride=2),
#                     Conv2d(12, 12, 3, 1, 1), nn.ReLU(),
#                     MaxPool2d(2, stride=2),
#                     nn.Flatten(),
#                     nn.Linear(12*7*7, 96), nn.ReLU(),
#                     nn.Linear(96, 48), nn.ReLU(),
#                     nn.Linear(48, 2))

LeNet = nn.Sequential(nn.Conv2d(3, 6, 5, 2, 2), nn.BatchNorm2d(6), nn.ReLU(),
                      nn.Conv2d(6, 12, 3, 1, 1), nn.BatchNorm2d(12), nn.ReLU(),
                      nn.MaxPool2d(2, stride=2),
                      nn.Conv2d(12, 24, 3, 1, 1), nn.BatchNorm2d(24), nn.ReLU(),
                      nn.MaxPool2d(2, stride=2),
                      nn.Conv2d(24, 48, 3, 1, 1), nn.BatchNorm2d(48), nn.ReLU(),
                      nn.MaxPool2d(2, stride=2),
                      nn.Conv2d(48, 96, 3, 1, 1), nn.BatchNorm2d(96), nn.ReLU(),
                      nn.MaxPool2d(2, stride=2),
                      nn.Flatten(),
                      nn.Linear(96 * 7 * 7, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                      nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                      nn.Linear(128, 2))

net_name = "LeNet"


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = leNet

    def forward(self, x):
        for net in self.net:
            x = forward(net, x)
        return x


__all__ = ["LeNet", "net_name"]
