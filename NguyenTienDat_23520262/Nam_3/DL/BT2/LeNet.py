import torch 
from torch import nn 
from torch.nn import functional as F

class LeNet(nn.Module): 
    def __init__(self, image_size, num_labels):
        super().__init__()
        self.w, self.h = image_size 
        self.input_size = self.w * self.h
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.avg_pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.FC1 = nn.Linear(16 * 5 * 5, 120)
        self.FC2 = nn.Linear(120, 84)
        self.FC3 = nn.Linear(84, num_labels)
        self.relu = nn.ReLU()
    def forward(self, x):
        """
            x: Tensor(batch_size, 28, 28)
        """
        x = x.unsqueeze(1)
        x = self.avg_pool(self.relu(self.conv1(x)))
        x = self.avg_pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5) # (B, 16*5*5)  
        x = self.relu(self.FC1(x))
        x = self.relu(self.FC2(x))
        x = self.FC3(x)
        return x