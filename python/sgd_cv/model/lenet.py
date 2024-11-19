import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LeNet5']

class LeNet5(nn.Module):
    '''
    参数设置参考Justin 598WI2022课件，非原lecun论文
    参数量相对更多一些
    
    Layer                           Output Size     Weight Size
    
    Input                           1 x 28 x 28
    Conv (Cout=20, K=5, P=2, S=1)   20 x 28 x 28    20 x 1 x 5 x 5
    ReLU                            20 x 28 x 28
    MaxPool(K=2, S=2)               20 x 14 x 14
    Conv (Cout=50, K=5, P=2, S=1)   50 x 14 x 14    50 x 20 x 5 x 5
    ReLU                            50 x 14 x 14
    MaxPool(K=2, S=2)               50 x 7 x 7
    Flatten                         2450
    Linear (2450 -> 500)            500             2450 x 500
    ReLU                            500
    Linear (500 -> 10)              10              500 x 10
    '''
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # 卷积层 1: 输入通道 1（灰度图），输出通道 20，卷积核大小 5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)  # 1x28x28 -> 20x28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # 卷积层 2: 输入通道 20，输出通道 50，卷积核大小 5x5
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=2)  # 20x14x14 -> 50x14x14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 50x14x14 -> 50x7x7

        # 全连接层 1
        self.fc1 = nn.Linear(50 * 7 * 7, 500)  # 50个 7x7 channel
        # 全连接层 2 (输出层)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        # conv 1 + 激活 + pooling
        x = self.pool1(F.relu(self.conv1(x)))
        # conv2 2 + 激活 + pooling
        x = self.pool2(F.relu(self.conv2(x)))

        x = torch.flatten(x, start_dim=1, end_dim=-1) # 注: dim-0 是Batch Size, 对单个样本展开
        # 全连接层 1 + 激活
        x = F.relu(self.fc1(x))
        # 全连接层 2 + 激活
        x = self.fc2(x)
        return x