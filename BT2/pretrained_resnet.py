import torch
from torch import nn
from transformers import ResNetForImageClassification
import torch.nn.functional as F

class PretrainedResnet(nn.Module):
    def __init__(self, num_classes=21, freeze_backbone=True):
        super().__init__()

        # Load pretrained ResNet
        basemodel = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.resnet = basemodel.resnet

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # New classifier head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize the new layers
        self._init_classifier()

    def _init_classifier(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, images: torch.Tensor):
        features = self.resnet(images).pooler_output
        features = features.squeeze(-1).squeeze(-1)
        logits = self.classifier(features)
        return logits
    
    def unfreeze_layers(self, num_layers=3):
        """Unfreeze the last n layers of the ResNet backbone"""
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        layers_to_unfreeze = list(self.resnet.named_parameters())[-num_layers:]
        for name, param in layers_to_unfreeze:
            param.requires_grad = True
            print(f"Unfroze layer: {name}")
