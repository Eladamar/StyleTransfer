import torch.nn as nn
import torchvision.models as models


class MyVgg(nn.Module):
    def __init__(self):
        super(MyVgg, self).__init__()
        self.model = models.vgg16(pretrained=True).features[:23]
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Store relevant features
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features


class MyVgg19(MyVgg):
  def __init__(self):
    # conv layers
    self.chosen_features = ["0", "5", "10", "19", "28"]
    # self.model = models.vgg19(pretrained=True).features[:29]
    super(MyVgg19, self).__init__()


class MyVgg16(MyVgg):
  def __init__(self):
    # ReLU layers
    self.chosen_features = ["3", "8", "15", "22"]
    super(MyVgg16, self).__init__()

