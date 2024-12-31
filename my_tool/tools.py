import torch.nn as nn


def print_net_shape(net, x, count=0):
    if isinstance(net, nn.Sequential):
        print("    "*count, "Sequential(", sep="")
        for deep_net in net:
            x = print_net_shape(deep_net, x, count+1)
        print(" "*10, "    " * count, ")", sep="")

    else:
        x = net(x)
        print(" " * 10 + "    " * count, end="", sep="")
        print(net.__class__.__name__, ":", x.shape)
    return x


def forward(net, x):
    if isinstance(net, nn.Sequential):
        for deep_net in net:
            x = forward(deep_net, x)
    else:
        x = net(x)
    return x
