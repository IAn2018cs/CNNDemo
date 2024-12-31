import torch.nn as nn
import torch
import sys
import os

# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.realpath(__file__))
# 将父目录添加到模块搜索路径中
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from my_tool.tools import forward, print_net_shape

__all__ = ["Vgg", "net_name"]


def vgg_block(num_conv, input_chanel, output_chanel):
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv2d(in_channels=input_chanel, out_channels=output_chanel, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(output_chanel))
        layers.append(nn.ReLU())
        input_chanel = output_chanel
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(input_chanel, conv_arch):
    # 前两个vgg块一层卷积，后三个两层卷积，一共8层卷积，接三个FC，一共11层，称呼为vgg-11
    conv_blk = []
    for (num_conv, out_chanel) in conv_arch:
        conv_blk.append(vgg_block(num_conv, input_chanel, out_chanel))
        input_chanel = out_chanel
    return nn.Sequential(
        *conv_blk,
        nn.Flatten(),
        nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, 2)
    )


conv_arch = ((1, 64), (1, 64), (2, 128), (2, 128))
vgg_net = vgg(3, conv_arch)
net_name = "vgg-11"


class Vgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = vgg_net

    def forward(self, x):
        for net in self.net:
            x = forward(net, x)
        return x


if __name__ == '__main__':
    x = torch.rand((64, 3, 64, 64))
    for net in vgg_net:
        x = print_net_shape(net, x)
