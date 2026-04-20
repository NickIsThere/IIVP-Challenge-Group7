import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = models.densenet121(num_classes=num_classes)

        # Adapt for 1-channel 32x32 images
        self.model.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Bypass the initial maxpool 
        self.model.features.pool0 = nn.Identity()

    def forward(self, x):
        return self.model(x)