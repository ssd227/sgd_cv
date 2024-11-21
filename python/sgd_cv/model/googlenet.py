'''
GoogleNet/Inception V1 V3

Note:
BatchNorm2d 和 RELU 不同时使用
    下面的实现，已经用BatchNorm替换掉了部分RELU
    有了BatchNorm，就不需要原论文多个分类头的trick，同时保证deep网络可以训练
    
    
todo
Inception v3实现
https://sh-tsang.medium.com/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c
https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GoogleNet']


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        """
        Args:
            in_channels: 输入通道数
            ch1x1: 1x1 卷积输出通道数
            ch3x3red: 3x3 卷积前的 1x1 降维通道数
            ch3x3: 3x3 卷积输出通道数
            ch5x5red: 5x5 卷积前的 1x1 降维通道数
            ch5x5: 5x5 卷积输出通道数
            pool_proj: 最大池化后的 1x1 卷积输出通道数
        """
        super(Inception, self).__init__()
        
        # 1x1 卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True)
        )
        
        # 1x1 降维 + 3x3 卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True)
        )
        
        # 1x1 降维 + 5x5 卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True)
        )
        
        # 最大池化 + 1x1 卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 并行分支计算并拼接
        outputs = [
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ]
        return torch.cat(outputs, dim=1)  # channel dim拼接


class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogleNet, self).__init__()
        
        # Stem network (aggressively downsamples input 224->56)
        self.stem = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            
            # maxpooling1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Conv2
            nn.Conv2d(64, 64, kernel_size=1), # Note: 1x1 conv, 混合channel信息
            nn.BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            
            # Conv3
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            
            # maxpooling2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception 模块
        self.inception3 = nn.Sequential(
            # inception3a
            Inception(192, 64, 96, 128, 16, 32, 32),
            #inception3b
            Inception(256, 128, 128, 192, 32, 96, 64),
        )
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception4 = nn.Sequential(
            # inception4a 
            Inception(480, 192, 96, 208, 16, 48, 64),
            # inception4b
            Inception(512, 160, 112, 224, 24, 64, 64),
            # inception4c 
            Inception(512, 128, 128, 256, 24, 64, 64),
            # inception4d 
            Inception(512, 112, 144, 288, 32, 64, 64),
            # inception4e
            Inception(528, 256, 160, 320, 32, 128, 128),
        )
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.inception5 = nn.Sequential(
            # inception5a
            Inception(832, 256, 160, 320, 32, 128, 128),
            #inception5b
            Inception(832, 384, 192, 384, 48, 128, 128),
        )
        
        # classifier (使用bn，删除aux1和aux2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), # kernel 压缩成1X1，channels作为features
            nn.Dropout(p=0.2, inplace=False),
            nn.Flatten(start_dim=1, end_dim=-1), # [1, 1024, 1, 1] -> [1, 1024]
            nn.Linear(in_features=1024, out_features=num_classes, bias=True),
        )
    
    def forward(self, x):
        x = self.stem(x)
        
        x = self.inception3(x)
        x = self.maxpool3(x)
        
        x = self.inception4(x)
        x = self.maxpool4(x)
        
        x = self.inception5(x) # [1, 1024, 7, 7]
                
        x = self.classifier(x)
        return x

# todo
# class InceptionV3(nn.Module):
