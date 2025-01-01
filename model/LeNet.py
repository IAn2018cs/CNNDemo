import torch.nn as nn

net_name = "LeNet"


class LeNet(nn.Module):
    def __init__(self, num_class=2):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),  # 224 -> 220
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 220 -> 110
            nn.Conv2d(6, 16, kernel_size=5),  # 110 -> 106
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 106 -> 53
        )
        self.flatten = nn.Flatten()
        # 假设输入尺寸为224x224，则此时应为16*53*53
        self.classifier = nn.Sequential(
            nn.Linear(16 * 53 * 53, 120),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(84, num_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


__all__ = ["LeNet", "net_name"]
