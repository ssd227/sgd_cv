'''
Mobile Net v1 and v2

cuda 直接爆内存了, 这个v2

模型理论上的优点,分分钟被DL框架实现的低效率给打败

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MobileNetV2'] # 目前只实现了V2 50层,参数差不多写死(参考torch实现)

# def Conv2dNormActivation(in_channels, out_channels, kernel_size, stride, padding, groups=1):
#     return nn.Sequential(
#             # conv1
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
#             nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # eps比googlente小
#             nn.ReLU6(inplace=True),
#             )
    
class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(Conv2dNormActivation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion): # expansion is t in lec pdf
        super(InvertedResidual, self).__init__()
        
        self.conv = nn.Sequential(
            # 1x1 Conv (Reduce the number of channels)
            Conv2dNormActivation(in_channels, in_channels*expansion, kernel_size=1, stride=1 , padding=0),
    
            # 3x3 Conv (Perform convolution)
            Conv2dNormActivation(in_channels*expansion, in_channels*expansion, kernel_size=3, stride=stride , padding=1, groups=in_channels * expansion), 

            # 1x1 Conv (Restore the number of channels)
            nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
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
    
class InvertedResidualShort(nn.Module):
    def __init__(self, in_channels, out_channels, expansion): # expansion is t in lec pdf
        super(InvertedResidualShort, self).__init__()
        
        self.conv = nn.Sequential(
            # 1x1 Conv (Reduce the number of channels)
            Conv2dNormActivation(in_channels, in_channels*expansion, kernel_size=3, stride=1 , padding=1, groups=32),

            # 1x1 Conv (Restore the number of channels)
            nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # 如果输入和输出通道不一致，需要匹配通道数
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
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


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, layers=None, t=None):
        super(MobileNetV2, self).__init__()
            
        # 定义四个残差块组，每组包含两个 BasicBlock
        self.features = nn.Sequential(
            Conv2dNormActivation(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1), # (0)
            InvertedResidualShort(in_channels=32, out_channels=16, expansion=1,), # (1)
            
            # 中间15层,全部是InvertedResidual
            InvertedResidual(in_channels=16, out_channels=24, stride=2, expansion=6,), # (2)
            InvertedResidual(in_channels=24, out_channels=24, stride=1, expansion=6,), # (3)
            
            InvertedResidual(in_channels=24, out_channels=32, stride=2, expansion=6,), # (4)
            InvertedResidual(in_channels=32, out_channels=32, stride=1, expansion=6,), # (5)
            InvertedResidual(in_channels=32, out_channels=32, stride=1, expansion=6,), # (6)
            
            InvertedResidual(in_channels=32, out_channels=64, stride=2, expansion=6,), # (7)
            InvertedResidual(in_channels=64, out_channels=64, stride=1, expansion=6,), # (8)
            InvertedResidual(in_channels=64, out_channels=64, stride=1, expansion=6,), # (9)
            InvertedResidual(in_channels=64, out_channels=64, stride=1, expansion=6,), # (10)
            
            InvertedResidual(in_channels=64, out_channels=96, stride=1, expansion=6,), # (11)
            InvertedResidual(in_channels=96, out_channels=96, stride=1, expansion=6,), # (12)
            InvertedResidual(in_channels=96, out_channels=96, stride=1, expansion=6,), # (13)
            
            InvertedResidual(in_channels=96, out_channels=160, stride=2, expansion=6,), # (14)           
            InvertedResidual(in_channels=160, out_channels=160, stride=1, expansion=6,), # (15)
            InvertedResidual(in_channels=160, out_channels=160, stride=1, expansion=6,), # (16)
            
            InvertedResidual(in_channels=160, out_channels=320, stride=1, expansion=6,), # (17)
            
            Conv2dNormActivation(in_channels=320, out_channels=1280, kernel_size=1, stride=1, padding=0), # (18)
            )

        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), # [1, 1280, 7, 7] -> [1, 1280, 1, 1]
            nn.Dropout(p=0.2, inplace=False),
            nn.Flatten(start_dim=1, end_dim=-1), # [1, 1280, 1, 1] -> [1, 1280]
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )

    # def _make_layer(self, in_channels, out_channels, num_blocks, stride, expansion):
    #     layers = []
    #     layers.append(InvertedResidual(in_channels, out_channels, stride, expansion))
    #     for _ in range(1, num_blocks):
    #         layers.append(InvertedResidual(out_channels, out_channels, stride=1, expansion=expansion))
    #     return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x) # 残差layers
        x = self.classifier(x) # 分类头
        return x
