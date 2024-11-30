'''
ResNet


按照架构有以下5种
# use Basic Block
resnet18 = models.resnet18()
resnet34 = models.resnet34()

# use Bottleneck Block
resnet50 = models.resnet50()
resnet101 = models.resnet101()
resnet152 = models.resnet152()

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']


# Stem network (aggressively downsamples input 224->56)
def Stem():
    return nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # eps比googlente小
            nn.ReLU(inplace=True),
            # pooling 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            )

# BasicBlock（for ResNet-18、ResNet-34）
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # conv2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
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


class ResNet(nn.Module):
    def __init__(self, layers, num_classes):
        super(ResNet, self).__init__()
        # stem
        self.stem = Stem() # 每次调用new一个新stem对象

        # 定义四个残差块组，每组包含两个 BasicBlock
        self.res_layers = nn.Sequential(
            self._make_layer(64, 64, layers[0], stride=1),
            self._make_layer(64, 128, layers[1], stride=2),
            self._make_layer(128, 256, layers[2], stride=2),
            self._make_layer(256, 512, layers[3], stride=2),
            )

        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), # kernel 压缩成1X1，channels作为features
            # nn.Dropout(p=0.2, inplace=False), # 参考torch实现，没加drop out
            nn.Flatten(start_dim=1, end_dim=-1), # [1, 512, 1, 1] -> [1, 512]
            nn.Linear(in_features=512, out_features=num_classes, bias=True)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)    # Stem network -- 缩小图片分辨率，增加channel  
        x = self.res_layers(x) # 残差layers
        x = self.classifier(x) # 分类头
        return x

    
def ResNet18(num_classes):
    return ResNet([2, 2, 2, 2], num_classes)

def ResNet34(num_classes):
    return ResNet([3, 4, 6, 3], num_classes)

##########################################################
##########################################################

# Bottleneck Block（for ResNet-50、ResNet-101、ResNet-152）
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        
        self.conv = nn.Sequential(
            # 1x1 Conv (Reduce the number of channels)
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True), # 第一次实验 怕不是就少这个relu就学不动？这么玄学
            
            # 3x3 Conv (Perform convolution)
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True), # 第一次实验 怕不是就少这个relu就学不动？这么玄学
            
            # 1x1 Conv (Restore the number of channels)
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True), # 第一次实验 怕不是就少这个relu就学不动？这么玄学
        )
        
        self.improving_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True), 
            
            # 1x1 Conv (Reduce the number of channels)
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True), 
            
            # 3x3 Conv (Perform convolution)
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1, bias=False),
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
        # out = self.conv(x)
        out = self.improving_conv(x)
        # 残差连接
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out


class ResNetDeep(nn.Module):
    def __init__(self, layers, num_classes):
        super(ResNetDeep, self).__init__()
        
        # stem
        self.stem = Stem() # 每次调用new一个新stem对象

        # 定义四个残差块组，每组包含两个 BasicBlock
        self.res_layers = nn.Sequential(
            self._make_layer(64, 256, layers[0], stride=1),
            self._make_layer(256, 512, layers[1], stride=2),
            self._make_layer(512, 1024, layers[2], stride=2),
            self._make_layer(1024, 2048, layers[3], stride=2),
            )

        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), # kernel 压缩成1X1，channels作为features
            # nn.Dropout(p=0.2, inplace=False), # 参考torch实现，没加drop out
            nn.Flatten(start_dim=1, end_dim=-1), # [1, 2048, 1, 1] -> [1, 2048]
            nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)    # Stem network -- 缩小图片分辨率，增加channel  
        x = self.res_layers(x) # 残差layers
        x = self.classifier(x) # 分类头
        return x

def ResNet50(num_classes):
    return ResNetDeep([3, 4, 6, 3], num_classes)

def ResNet101(num_classes):
    return ResNetDeep([3, 4, 23, 3], num_classes)

def ResNet152(num_classes):
    return ResNetDeep([3, 8, 36, 3], num_classes)


# ------------------- 待删除 -------------------
'''
class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        # stem
        self.stem = Stem() # 每次调用new一个新stem对象
        
        # 定义四个残差块组，每组包含两个 BasicBlock
        self.res_layers = nn.Sequential(
            self._make_layer(64, 64, 3, stride=1),
            self._make_layer(64, 128, 4, stride=2),
            self._make_layer(128, 256, 6, stride=2),
            self._make_layer(256, 512, 3, stride=2),
            )

        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), # kernel 压缩成1X1，channels作为features
            # nn.Dropout(p=0.2, inplace=False), # 参考torch实现，没加drop out
            nn.Flatten(start_dim=1, end_dim=-1), # [1, 512, 1, 1] -> [1, 512]
            nn.Linear(in_features=512, out_features=num_classes, bias=True)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)    # Stem network -- 缩小图片分辨率，增加channel  
        x = self.res_layers(x) # 残差layers
        x = self.classifier(x) # 分类头
        return x

'''
