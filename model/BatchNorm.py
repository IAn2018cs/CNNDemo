import torch
import torch.nn as nn


def batch_norm(x, gamma, beta, moving_mean, moving_var, eps=1e-5, momentum=0.9):
    # 如果在预测模式，就使用moving
    if not torch.is_grad_enabled():
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(x.shape) in (2, 4)
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + eps)
        # 更新移动的均值和方差
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * moving_var
        # 缩放和移动位置
    y = gamma * x_hat + beta

    return y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    # num_dim 2是MLP 4是Conv
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与梯度和迭代拉伸的参数，初始化为1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, x):
        # 如果x不在内存上，将moving_mean和moving_var复制上去
        if self.moving_mean.device != x.device:
            self.moving_mean.device = x.device
            self.moving_var.device = x.device

        y, self.moving_mean, self.moving_var = batch_norm(x, self.gamma, self.beta,
                                                          self.moving_mean, self.moving_var, )

        return y


if __name__ == '__main__':
    x = torch.rand((64, 3, 64, 64))
    batch_1 = BatchNorm(3, 4)
    print(batch_1(x).shape)
