import torch.nn as nn
from Unet import UNet

class ImageTransformUNet(nn.Module):
    def __init__(self):
        super(ImageTransformUNet, self).__init__()
        self.model = UNet()

    def forward(self, x):
        out = self.model(x)
        return out
