import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['VGG16', 'VGG19']

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_convs: 该块包含的卷积层数量
        """
        super(ConvBlock, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels  # update输入通道数
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # pooling
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 64, num_convs=2),   # Block 1
            ConvBlock(64, 128, num_convs=2), # Block 2
            ConvBlock(128, 256, num_convs=3), # Block 3
            ConvBlock(256, 512, num_convs=3), # Block 4
            ConvBlock(512, 512, num_convs=3)  # Block 5
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            # nn.Flatten(),
            # 全连接层 1
            nn.Linear(512*7*7, 4096, bias=True),  # 512个 7x7 channel = 25088
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            # 全连接层 2
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            # 全连接层 3 (输出层)
            nn.Linear(4096, num_classes, bias=True),
        )
    
    def forward(self, x):
        # conv & pooling
        x = self.features(x)
        
        # 同时处理不同size的图片
        # x = self.avgpool(x)
        
        # flatten操作
        x = torch.flatten(x, start_dim=1, end_dim=-1) # 注: dim-0 是Batch Size, 对单个样本展开
        
        # MLP分类
        x= self.classifier(x)

        return x


class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 64, num_convs=2),   # Stage 1
            ConvBlock(64, 128, num_convs=2), # Stage 2
            ConvBlock(128, 256, num_convs=3), # Stage 3
            ConvBlock(256, 512, num_convs=4), # Stage 4
            ConvBlock(512, 512, num_convs=4)  # Stage 5
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            # nn.Flatten(),
            # 全连接层 1
            nn.Linear(512*7*7, 4096, bias=True),  # 512个 7x7 channel = 25088
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            # 全连接层 2
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            # 全连接层 3 (输出层)
            nn.Linear(4096, num_classes, bias=True),
        )
    
    def forward(self, x):        
        # conv & pooling
        x = self.features(x)
        
        # 同时处理不同size的图片
        # x = self.avgpool(x)
        
        # flatten操作
        x = torch.flatten(x, start_dim=1, end_dim=-1) # 注: dim-0 是Batch Size, 对单个样本展开

        # MLP分类
        x= self.classifier(x)

        return x
