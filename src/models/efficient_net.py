import torch.nn as nn
import torchvision.models as models

class EfficientNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # No pre-learned weights
        self.model = models.efficientnet_b0(weights=None)

        # Adapt for 1 channel
        self.model.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Final classifier with output of 1280 features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), # Use dropout to avoid overfitting
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        return self.model(x)