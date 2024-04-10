import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BubbleNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(BubbleNet, self).__init__()
        # Load pretrained MobileNetV3 Large
        mobilenet_v3_large = models.mobilenet_v3_large(pretrained=pretrained)
        # Remove the classifier
        self.mobilenet_encoder = nn.Sequential(*list(mobilenet_v3_large.features.children()))

        # Assuming the output feature size of MobileNetV3 Large is 960 for the last block
        self.decoder = nn.Sequential(
            # Start from the output of the last block, assuming it maintains spatial dimensions
            nn.ConvTranspose2d(960, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, num_classes, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        # Encoder
        x = self.mobilenet_encoder(x)  # Removed the indexing since we are using features only
        
        # Check the shape here and make sure it's 4D

        # Decoder
        x = self.decoder(x)
        
        # Ensure output is the same size as input
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        return x