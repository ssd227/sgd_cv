import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AlexNet']

class AlexNet(nn.Module):
    '''
    参数设置参考Justin 598WI2022课件
    
            Input size      Layer                       Output size
    Layer   C       H / W   filters kernel  stride pad  C       H / W   memory (KB) params (k) flop (M)
    conv1   3       227     64      11      4       2   64      56      784         23          73
    pool1   64      56              3       2       0   64      27      182         0           0
    conv2   64      27      192     5       1       2   192     27      547         307         224
    pool2   192     27              3       2       0   192     13      127         0           0
    conv3   192     13      384     3       1       1   384     13      254         664         112
    conv4   384     13      256     3       1       1   256     13      169         885         145
    conv5   256     13      256     3       1       1   256     13      169         590         100
    pool5   256     13              3       2       0   256     6       36          0           0
    flatten 256     6                                   9216            36          0           0
    fc6     9216            4096                        4096            16          37,749      38
    fc7     4096            4096                        4096            16          16,777      17
    fc8     4096            1000                        1000            4           4,096       4
    '''
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # Conv 1: 输入通道 3，输出通道 64，卷积核大小 11x11
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),  # 3x28x28 -> 64x56x56
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 64x56x56 -> 64x27x27

            # Conv 2: 输入通道 64，输出通道 192，卷积核大小 5x5
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2),  # 64x27x27 -> 192x27x27
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 192x27x27 -> 192x13x13
            
            # Conv 3: 输入通道 192，输出通道 384，卷积核大小 3x3
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),  # 192x13x13 -> 256x13x13
            nn.ReLU(),
            
            # Conv 4: 输入通道 384，输出通道 256，卷积核大小 3x3
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # 256x13x13 -> 256x13x13
            nn.ReLU(),
            
            # Conv 5: 输入通道 256，输出通道 256，卷积核大小 3x3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 256x13x13 -> 256x13x13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256x13x13 -> 256x6x6

        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6)) # torch官方实现，同时处理不同size的图片
        
        self.classifier = nn.Sequential(   
            # 全连接层 1
            nn.Dropout(p=0.5),
            nn.Linear(256*6*6, 4096, bias=True),  # 256个 6x6 channel = 9216
            nn.ReLU(),
            
            # 全连接层 2
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(),
            
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