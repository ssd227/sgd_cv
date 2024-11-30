'''
ResNeXt


'''
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ResNeXt50', 'ResNeXt101']


# Stem network (aggressively downsamples input 224->56)
def Stem():
    return nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # eps比googlente小
            nn.ReLU(inplace=True),
            # pooling 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            )

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups,):
        super(Block, self).__init__()
        
        '''
        TODO
            在不放缩4C->Gc的前提下, 只修改中间conv训练速度并不能提升多少
                FLOPs 和训练速度并不等价, 需要考虑训练框架底层优化逻辑
        '''
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # 1x1 Conv (Reduce the number of channels)
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True), 

            # 3x3 Conv (Perform convolution)
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            
            # 1x1 Conv (Restore the number of channels)
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, bias=False),
        )
        
        # 如果输入和输出通道不一致，需要匹配通道数
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv(x)
        # 残差连接
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out

class ResNeXt(nn.Module):
    def __init__(self, layers, groups, num_classes):
        super(ResNeXt, self).__init__()
        # stem
        self.stem = Stem()

        # 定义四个残差块组，每组包含两个 BasicBlock
        self.res_layers = nn.Sequential(
            self._make_layer(64, 256, layers[0], stride=1, groups=groups),
            self._make_layer(256, 512, layers[1], stride=2, groups=groups),
            self._make_layer(512, 1024, layers[2], stride=2, groups=groups),
            self._make_layer(1024, 2048, layers[3], stride=2, groups=groups),
            )

        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), # kernel 压缩成1X1，channels作为features
            # nn.Dropout(p=0.2, inplace=False), # 参考torch实现，没加drop out
            nn.Flatten(start_dim=1, end_dim=-1), # [1, 2048, 1, 1] -> [1, 2048]
            nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, groups):
        layers = []
        layers.append(Block(in_channels, out_channels, stride, groups))
        for _ in range(1, num_blocks):
            layers.append(Block(out_channels, out_channels, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)    # 缩小图片分辨率，增加channel  
        x = self.res_layers(x) # 残差layers
        x = self.classifier(x) # 分类
        return x


def ResNeXt50(num_classes):
    return ResNeXt([3, 4, 6, 3], 32, num_classes)

def ResNeXt101(num_classes):
    return ResNeXt([3, 4, 23, 3], 32, num_classes)

