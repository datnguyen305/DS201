import torch 
from torch import nn 
from torch.nn import functional as F

class GoogLeNet(nn.Module):
    def __init__(self, image_size, num_labels):
        super().__init__()
        self.c, self.h, self.w = image_size
        self.Conv7_1 = nn.Conv2d(
            in_channels=self.c,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.Maxpool1_2 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0,
            ceil_mode=True
        )
        self.Conv1_3 = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.Maxpool3_4 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.Maxpool3_7 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.Maxpool3_13 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(0.4)
        self.Linear = nn.Linear(1024, num_labels)
        # === Inception blocks ===
        self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)   # output: 256
        self.inception_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64) # output: 480
        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64) # output: 512
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64) # output: 512
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64) # output: 512
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64) # output: 528
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128) # output: 832
        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128) # output: 832
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128) # output: 1024
    def forward (self, x):
        # x : Tensor(B, 3, 224, 224)
        x = F.relu(self.Conv7_1(x))   # (B, 64, H/2, W/2)
        x = self.Maxpool1_2(x)        # (B, 64, H/4, W/4)
        x = F.relu(self.Conv1_3(x))   # (B, 192, H/4, W/4)
        x = self.Maxpool3_4(x)        # (B, 192, H/8, W/8)

        x = self.inception_3a(x)      # (B, 256, H/8, W/8)
        x = self.inception_3b(x)      # (B, 480, H/8, W/8)
        x = self.Maxpool3_7(x)        # (B, 480, H/16, W/16)

        x = self.inception_4a(x)      # (B, 512, H/16, W/16)
        x = self.inception_4b(x)      # (B, 512, H/16, W/16)
        x = self.inception_4c(x)      # (B, 512, H/16, W/16)
        x = self.inception_4d(x)      # (B, 528, H/16, W/16)
        x = self.inception_4e(x)      # (B, 832, H/16, W/16)
        x = self.Maxpool3_13(x)       # (B, 832, H/32, W/32)

        x = self.inception_5a(x)      # (B, 832, H/32, W/32)
        x = self.inception_5b(x)      # (B, 1024, H/32, W/32)

        x = self.Avgpool(x)           # (B, 1024, H/64, W/64)
        x = torch.flatten(x, start_dim=1)  # (B, 1024)
        x = self.Dropout(x)
        x = self.Linear(x)            # (B, num_labels)
        return x
    
class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        # 1x1 branch
        out_1x1: int,
        # 1x1 -> 3x3 branch
        red_3x3: int,
        out_3x3: int,
        # 1x1 -> 5x5 branch
        red_5x5: int,
        out_5x5: int,
        # pool -> 1x1 branch
        pool_proj: int,
    ):
        super().__init__()

        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 conv → 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 conv → 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        # Branch 4: 3x3 maxpool → 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ]
        return torch.cat(outputs, dim=1)

