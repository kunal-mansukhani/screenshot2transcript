import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch.nn.init as init

class BubbleNet(nn.Module):
    def __init__(self, num_classes=3):
        super(BubbleNet, self).__init__()
        # Load pretrained MobileNetV3 Large
        mobilenet_v3_large = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        # Remove the classifier
        self.mobilenet_encoder = nn.Sequential(*list(mobilenet_v3_large.features.children()))

        self.decoder = nn.Sequential(
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

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.mobilenet_encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x

    def _initialize_weights(self):
        # Initialize weights of decoder
        for m in self.decoder.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
