import torch 
from torch import nn 
from torch.nn import functional as F

class ResNet18(nn.Module):
    def __init__(self, image_size, num_labels):
        super().__init__()
        self.c, self.h, self.w = image_size
        self.conv1 = nn.Conv2d(
            in_channels=self.c,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        # Define ResNet Blocks
        self.layer1 = nn.Sequential(
            ResNetBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ResNetBlock(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(64, 128, kernel_size=3, stride=2, padding=1, conv=True, padding_identity=0, stride_identity=2),
            ResNetBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ResNetBlock(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(128, 256, kernel_size=3, stride=2, padding=1, conv=True, padding_identity=0, stride_identity=2),
            ResNetBlock(256, 256, kernel_size=3, stride=1, padding=1),
        )
        self.layer4 = nn.Sequential(
            ResNetBlock(256, 512, kernel_size=3, stride=2, padding=1, conv=True, padding_identity=0, stride_identity=2),
            ResNetBlock(512, 512, kernel_size=3, stride=1, padding=1),
        ) 

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_labels)
    
    def forward(self, x):
        # x: Tensor(B, C, H, W)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, H/2, W/2)
        x = self.maxpool(x)                   # (B, 64, H/4, W/4)
        x = self.layer1(x)                    # (B, 64, H/4, W/4)
        x = self.layer2(x)                    # (B, 128, H/8, W/8)
        x = self.layer3(x)                    # (B, 256, H/16, W/16)
        x = self.layer4(x)                    # (B, 512, H/32, W/32)
        x = self.avgpool(x)                   # (B, 512, 1, 1)
        x = torch.flatten(x, 1)               # (B, 512)
        x = self.fc(x)                        # (B, num_labels)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, conv=False, padding_identity=0, stride_identity=1):
        super().__init__()
        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution layer (stride should be 1)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,  # Always 1 for conv2
            padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.use_shortcut = conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride_identity,
                padding=padding_identity
            ),
            nn.BatchNorm2d(out_channels)
        ) if conv else None

    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut connection
        if self.use_shortcut:
            identity = self.shortcut(x)
            
        # Add shortcut
        out += identity
        out = F.relu(out)
        
        return out