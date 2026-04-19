import torch.nn as nn
import torchvision.models as models

class ResNetCustom(nn.Module):
    def __init__(self, model_name='ResNet18', num_classes=10):
        super().__init__()
        
        # Load the base model without pretrained weights
        if model_name == 'ResNet18':
            self.model = models.resnet18(num_classes=num_classes)
        elif model_name == 'ResNet34':
            self.model = models.resnet34(num_classes=num_classes)
        else:
            raise ValueError("Unsupported ResNet type")
            
        # Adapt for 1-channel 32x32 images
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        #Bypass the initial maxpool 
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)
