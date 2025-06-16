import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        # stride=2 路徑
        self.conv1_s2 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0)
        self.bn1_s2 = nn.BatchNorm2d(32)
        self.conv2_s2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.bn2_s2 = nn.BatchNorm2d(64)
        # 計算 flatten 後的維度
        self.flatten_dim = 64 * ((img_size - 2) // 4) * ((img_size - 2) // 4)
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # stride=1 路徑
        x1 = F.silu(self.bn1(self.conv1(x)))         #  1x28x28 -> 32x26x26
        x1 = F.max_pool2d(x1, kernel_size=2)         #  32x26x26 -> 32x13x13
        x1 = F.silu(self.bn2(self.conv2(x1)))        #  32x13x13 -> 64x11x11
        x1 = F.max_pool2d(x1, kernel_size=2)         #  64x11x11 -> 64x5x5
        x1 = F.pad(x1, (0, 1, 0, 1))             # 64x5x5 -> 64x6x6

        # stride=2 路徑
        x2 = F.silu(self.bn1_s2(self.conv1_s2(x)))   # 1x28x28 -> 32x13x13
        x2 = F.silu(self.bn2_s2(self.conv2_s2(x2)))  # 32x13x13 -> 64x6x6
        # 將兩路徑結果相加
        x_sum = x1 + x2
        x = x_sum.view(x_sum.size(0), -1)
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
