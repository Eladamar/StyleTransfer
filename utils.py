import torch
from PIL import Image


def load_image(image_name, transformer=None):
    image = Image.open(image_name)
    if transformer:
      image = transformer(image)
      image = image.unsqueeze(0)
    return image


def batch_gram(image):
    b, c, h, w = image.size()
    f = image.view(b, c, -1)
    f_t = f.transpose(1, 2)
    G = torch.div(f.bmm(f_t), c * h * w)
    return G

def batch_loss(input, target, criterion):
  if input.shape == target.shape:
    return criterion(input, target)

  loss = 0
  for t in target.split(1):
    loss += criterion(input, t)
  return loss