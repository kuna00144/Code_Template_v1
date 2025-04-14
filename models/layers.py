import torch
import torch.nn as nn
from torch.nn import init

class VGG16(nn.Module):
    def __init__(self, in_channels=1, out_channels=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, out_channels)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, mid_channels, stride = 1, projection = None):
        super().__init__()
        
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels * self.expansion, 3, padding = 1, bias = False),
            nn.BatchNorm2d(mid_channels * self.expansion)
        )
        
        self.projection = projection
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        residual = self.residual(x)
        
        if self.projection is not None:
            skip_connection = self.projection(x)
        else:
            skip_connection = x
            
        out = self.ReLU(residual + skip_connection)
        return out
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, mid_channels, stride = 1, projection = None):
        super().__init__()
        
        self.residual = nn.Sequential( # 1x1 -> 3x3 -> 1x1 구조로 연산량 줄임
            nn.Conv2d(in_channels, mid_channels, 1, stride = stride, bias = False), # 1x1
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding = 1, bias = False), # 3x3
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels * self.expansion, 1, bias = False), # 1x1
            nn.BatchNorm2d(mid_channels * self.expansion)
        )
        
        self.projection = projection
        self.ReLU = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        
        if self.projection is not None:
            skip_connection = self.projection(x)
        else:
            skip_connection = x
            
        out = self.ReLU(residual + skip_connection)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_block_list, n_classes = 1000):
        super().__init__()
        assert len(num_block_list) == 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(3, stride = 2, padding = 1)
        
        self.in_channels = 64
        self.stage1 = self.make_stage(block, 64, num_block_list[0], stride = 1)
        self.stage2 = self.make_stage(block, 128, num_block_list[1], stride = 2)
        self.stage3 = self.make_stage(block, 256, num_block_list[2], stride = 2)
        self.stage4 = self.make_stage(block, 512, num_block_list[3], stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, n_classes)
    
    def make_stage(self, block, mid_channels, num_blocks,  stride = 1):
        if stride != 1 or self.in_channels != mid_channels * block.expansion:
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, mid_channels * block.expansion, 1, stride = stride, bias = False),
                nn.BatchNorm2d(mid_channels * block.expansion)
            )
        else:
            projection = None
        
        layers = []
        for idx in range(num_blocks):
            if idx == 0:
                layers.append(block(self.in_channels, mid_channels, stride, projection))
                self.in_channels = mid_channels * block.expansion
            else:
                layers.append(block(self.in_channels, mid_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)