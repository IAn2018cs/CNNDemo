import torch
import torch.nn as nn


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(X.shape[0] - h + 1):
        for j in range(X.shape[1] - w + 1):
            Y[i, j] = (X[i: i + h, j:j + w] * K).sum()
    return Y


# 定义卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

    def backward(self, lr=5e-3):
        for name, param in self.named_parameters():
            if isinstance(param, nn.Parameter):
                if param.requires_grad:
                    param.data -= lr * param.grad


# 定义边缘填充
def padding_corr2d(X, K, padding):
    new_shape = (X.shape[0] + padding[0] * 2, X.shape[1] + padding[1] * 2)
    temp = torch.zeros(new_shape)
    temp[padding[0]: new_shape[0] - padding[0], padding[1]: new_shape[1] - padding[1]] = X
    Y = corr2d(temp, K)
    return Y


def padding_stride_corr2d(X, K, padding, stride):
    new_shape = (X.shape[0] + padding[0] * 2, X.shape[1] + padding[1] * 2)
    temp = torch.zeros(new_shape)
    temp[padding[0]: new_shape[0] - padding[0], padding[1]: new_shape[1] - padding[1]] = X
    n_h, n_w = temp.shape
    k_h, k_w = K.shape

    new_row = int((n_h - k_h) / stride[0] + 1)
    new_column = int((n_w - k_w) / stride[1] + 1)

    if not ((n_h - k_h) / stride[0] + 1).is_integer() or not ((n_w - k_w) / stride[1] + 1).is_integer():
        print("can not reach the end")

    Y = torch.zeros(new_row, new_column)

    for i in range(new_row):
        for j in range(new_column):
            Y[i, j] = (temp[i * stride[0]: i * stride[0] + k_h, j * stride[1]: j * stride[1] + k_w] * K).sum()
    return Y


def corr2d_multi_in(X, K, padding=(0, 0), stride=(1, 1)):
    Y = sum(padding_stride_corr2d(x, k, padding, stride) for x, k in zip(X, K))
    print(f"corr2d_multi_in: X: {X.shape} K: {K.shape} Y: {Y.shape}")
    return Y


def corr2d_multi_in_out(X, K, padding=(0, 0), stride=(1, 1)):
    Y = torch.stack([corr2d_multi_in(X, k, padding, stride) for k in K], dim=0)
    print(f"corr2d_multi_in_out: X: {X.shape} K: {K.shape} Y: {Y.shape}")
    return Y


def corr2d_multi_in_out_1x1(X, K):
    c_in, h, w = X.shape
    c_out = K.shape[0]
    X = X.reshape(c_in, h * w)
    K = K.reshape(c_out, c_in)
    Y = torch.matmul(K, X).reshape(c_out, h, w)
    print(f"corr2d_multi_in_out_1x1: X: {X.shape} K: {K.shape} Y: {Y.shape}")
    return


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for r in range(Y.shape[0]):
        for c in range(Y.shape[1]):
            if mode == 'max':
                Y[r, c] = X[r: r + p_h, c: c + p_w].max()
            elif mode == 'mean':
                Y[r, c] = X[r: r + p_h, c: c + p_w].mean()
            else:
                raise Exception("You can only choose max or mean pooling function")
    return Y


def pool2d_padding_stride_multi_in(X, pool_size, padding=(0, 0), stride=(1, 1), mode="max"):
    batch_size, c_in, h, w = X.shape
    h, w = h + padding[0] * 2, w + padding[1] * 2
    p_h, p_w = pool_size
    Y = torch.zeros((batch_size, c_in, int((h - p_h) / stride[0] + 1), int((w - p_w) / stride[1] + 1)))
    for i in range(batch_size):
        for r in range(Y.shape[-2]):
            for c in range(Y.shape[-1]):
                if mode == "max":
                    Y[i, :, r, c] = torch.stack(
                        [x[r * stride[0]: r * stride[0] + p_h, c * stride[1]: c * stride[1] + p_w].max() for x in X[i]])
                elif mode == "mean":
                    Y[i, :, r, c] = torch.stack(
                        [x[r * stride[0]: r * stride[0] + p_h, c * stride[1]: c * stride[1] + p_w].mean() for x in
                         X[i]])
                else:
                    raise Exception("Can only choose max or mean pooling method")
    print("Pooling size: ", Y.shape)
    return Y


class Conv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, padding=0, stride=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.c_out = c_out
        self.device = torch.device('cpu')

        self.weights_k = nn.Parameter(torch.normal(0, 1, (c_out, c_in, *kernel_size)), requires_grad=True)
        self.weights_b = nn.Parameter(torch.zeros(c_out, c_in, 1), requires_grad=True)

    def forward(self, X: torch.tensor):
        batch_size, c_in, h, w = X.shape
        y_h, y_w = (int((h + self.padding[0] * 2 - self.kernel_size[0]) / self.stride[0]) + 1,
                    int((w + self.padding[1] * 2 - self.kernel_size[1]) / self.stride[1]) + 1)

        Y = torch.zeros((batch_size, self.c_out, y_h, y_w)).to(self.device)
        for batch_index in range(batch_size):
            Y[batch_index, :, :, :] = self.corr2d_multi_in_out(X[batch_index], self.weights_k, self.weights_b,
                                                               padding=self.padding, stride=self.stride)
        return Y

    def corr2d_padding_strid(self, X, K, B, padding, stride):
        """
        :param X:  输入的图片，大小为hxw
        :param K:  输入的卷积核，大小为k_h, k_w
        :param B:  输入的骗纸项，大小为 1
        :param padding: 边缘补充的大小:tuple
        :param stride:  步长:tuple
        :return: 卷积后的结果，大小为(h-k_h + padding) / stride + 1 x (w-k_w + padding) / stride
        """
        h, w = X.shape
        k_h, k_w = K.shape
        Y = torch.zeros(
            (int((h - k_h + padding[0] * 2) / stride[0]) + 1, int((w - k_w + padding[1] * 2) / stride[1]) + 1)).to(
            self.device)
        temp = torch.zeros((h + padding[0] * 2, w + padding[1] * 2))
        temp[padding[0]: padding[0] + h, padding[1]: padding[1] + w] = X
        X = temp.to(self.device)

        for r in range(Y.shape[0]):
            for c in range(Y.shape[1]):
                Y[r, c] = (X[r * stride[0]: r * stride[0] + k_h, c * stride[1]: c * stride[1] + k_w] * K).sum()
        return Y + B

    def corr2d_multi_in(self, X, K, B, padding, stride):
        """
        多通道输入
        :param X: size: c_in x h x w
        :param K: c_in x k_h x k_w
        :param B: c_in x 1
        :param padding: 边缘填充 tuple
        :param stride: 步长 tuple
        :return: h_y x w_y
        """
        if X.shape[0] != K.shape[0]:
            raise Exception("input channels error")
        return sum(self.corr2d_padding_strid(x, k, b, padding, stride) for x, k, b in zip(X, K, B))

    def corr2d_multi_in_out(self, X, K, B, padding, stride):
        """
        多通道输出
        :param X: c_in x h x w
        :param K: c_out x c_in x k_h x k_w
        :param B: c_out x c_in x 1
        :param padding: 填充 tuple
        :param stride:  步长 stride
        :return: Y.shape c_out x h_y x w_y
        """

        Y = torch.stack([self.corr2d_multi_in(X, k, b, padding, stride) for k, b in zip(K, B)], dim=0)
        return Y

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, padding=0, stride=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.device = torch.device('cpu')

    def forward(self, X):
        batch_size, c_in, h, w = X.shape
        Y = torch.zeros((batch_size, c_in, int((h - self.kernel_size[0] + self.padding[0] * 2) / self.stride[0]) + 1,
                         int((w - self.kernel_size[1] + self.padding[1] * 2) / self.stride[1]) + 1))
        for r in range(Y.shape[-2]):
            for c in range(Y.shape[-1]):
                max_value1, _ = torch.max(X[:, :, r * self.stride[0]: r * self.stride[0] + self.kernel_size[0],
                                          c * self.stride[1]: c * self.stride[1] + self.kernel_size[1]], dim=-1)
                max_value2, _ = torch.max(max_value1, dim=-1)
                Y[:, :, r, c] = max_value2
        return Y.to(self.device)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]


if __name__ == '__main__':
    "first try -- 相关性的计算，卷积过程"
    # X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    # K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    # print(corr2d(X, K))

    "second try -- 区分边界"
    # X = torch.ones((6, 8))
    # X[:, 2:6] = 0
    # K = torch.tensor([[1.0, -1.0]])
    # Y = corr2d(X, K)
    # print(f"X: {X}", f"K: {K}", f"Y: {Y}", sep="\n")

    "third try -- 转置矩阵看看效果"
    # X = torch.ones((6, 8))
    # X[:, 2:6] = 0
    # K = torch.tensor([[1.0, -1.0]])
    # X = X.t()
    # Y = corr2d(X, K)
    # print(f"X: {X}", f"K: {K}", f"Y: {Y}", sep="\n")

    "fourth try -- 学习卷积核"
    # X = torch.ones((6, 8))
    # X[:, 2:6] = 0
    # K = torch.tensor([[1.0, -1.0]])
    # Y = corr2d(X, K)
    #
    # conv2d = Conv2D((1, 2))
    # for i in range(100):
    #     Y_hat = conv2d(X)
    #     loss = (Y - Y_hat)**2
    #
    #     conv2d.zero_grad()
    #     loss.sum().backward()
    #
    #     conv2d.backward()
    #
    #     if i % 10 == 0:
    #         print(f"Epoch； {i}, Loss: {loss.sum():.4f}")
    #
    # print(conv2d.weight.data)
    # print(conv2d.bias.data)

    "fifth try -- 边缘填充padding"
    # X = torch.ones((6, 8))
    # K = torch.rand((3, 3))
    # print(padding_corr2d(X, K, padding=(1, 1)).shape)

    "sixth try -- 边缘填充加步长"
    # X = torch.ones((8, 8))
    # K = torch.rand((3, 3))
    # print(padding_stride_corr2d(X, K, padding=(1, 1), stride=(2, 2)).shape)

    "seven try -- 多通道输入卷积"
    # X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
    #                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    # K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    # corr2d_multi_in(X, K)

    "8 try -- 多通道输出"
    # X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
    #                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    # K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    # K = torch.stack((K, K+1, K+2), 0)
    # corr2d_multi_in_out(X, K)

    "9 try -- 1x1的卷积核"
    # X = torch.normal(0, 1, (3, 3, 3))
    # K = torch.normal(0, 1, (2, 3, 1, 1))
    # corr2d_multi_in_out(X, K)
    # corr2d_multi_in_out_1x1(X, K)

    "10 try --- 池化操作"
    # X = torch.arange(0, 9, dtype=torch.float).view(3, 3)
    # print(pool2d(X, (2, 2), mode="mean"))

    "11 try -- 池化操作，padding和stride设置，多通道输入"
    # X = torch.arange(64, dtype=torch.float).view(2, 2, 4, 4)
    # print(X)
    # print(pool2d_padding_stride_multi_in(X, (2, 2)))

    "12 try -- 卷积神经网络定义"
    # X = torch.rand((6, 6, 28, 28))
    # conv2d = Conv2d(6, 6, 5, 2)
    # X = conv2d(X)
    # print(conv2d.__class__.__name__, ":", X.shape)
    # maxpool2d = MaxPool2d(kernel_size=2, stride=2)
    # print(maxpool2d.__class__.__name__, ":", maxpool2d(X).shape)

    "13 try -- 搭建经典卷积网络 leNet"
    leNet = nn.Sequential(Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                          MaxPool2d(kernel_size=2, stride=2),
                          Conv2d(6, 16, kernel_size=5, padding=0),
                          MaxPool2d(kernel_size=2, stride=2),
                          nn.Flatten(),
                          nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                          nn.Linear(120, 84), nn.Sigmoid(),
                          nn.Linear(84, 10), nn.Softmax(dim=1))
    X = torch.rand(size=(2, 1, 28, 28), dtype=torch.float32)
    print(leNet)
    for layer in leNet:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)
    print(X)
