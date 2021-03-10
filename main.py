import os

# import torch
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from datetime import datetime

from Unet import UNet
from VGG import MyVgg16, MyVgg19
from utils import *
from train import Trainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.001
batch_size = 8
imsize = 256

# Root directory of the dataset
data_root = '/content/data'
style_folder = '/content/drive/MyDrive/StyleTransfer/styles'
if not os.path.isdir(data_root):
  raise Exception(f'No such folder {data_root}')

transformer = transforms.Compose(
    [
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# load data

styles = ImageFolder(style_folder, transformer)
print(styles.imgs)
y_s = torch.stack([i for i,_ in ImageFolder(style_folder, transformer)])



dataset = ImageFolder(data_root, transformer)
# dataset = torch.utils.data.Subset(dataset, range(30000))
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, drop_last=True)

# models
VGG = MyVgg16()
image_transformer = UNetSmall(batch_size)

optimizer = optim.Adam(image_transformer.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()
log_name = f'{type(image_transformer).__name__}_{datetime.now().strftime("%m%d-%H")}'

trainer = TrainerMul(image_transformer=image_transformer,
                vgg=VGG,
                optimizer=optimizer,
                criterion=criterion,
                style=y_s,
                device=device,
                tensorboard_logger=log_name)

trainer.train(data_loader, 2)


