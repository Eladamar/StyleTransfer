import os
import argparse

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

def main(ARGS):

    data_root = ARGS.data_root
    if not os.path.isdir(data_root):
        raise Exception(f'No such folder {data_root}')

    transformer = transforms.Compose(
        [
            transforms.Resize((ARGS.image_size, ARGS.image_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # load data
    styles = ImageFolder(ARGS.style_folder, transformer)
    print(styles.imgs)
    y_s = torch.stack([i for i, _ in styles])

    dataset = ImageFolder(data_root, transformer)
    # dataset = torch.utils.data.Subset(dataset, range(30000))
    data_loader = DataLoader(dataset=dataset, batch_size=ARGS.batch_size, num_workers=8, drop_last=True)

    # models
    VGG = MyVgg16()
    image_transformer = UNet(len(styles))

    optimizer = optim.Adam(image_transformer.parameters(), lr=ARGS.learning_rate)
    criterion = torch.nn.MSELoss()
    log_name = f'{type(image_transformer).__name__}_{datetime.now().strftime("%m%d-%H")}'

    trainer = Trainer(image_transformer=image_transformer,
                      vgg=VGG,
                      optimizer=optimizer,
                      criterion=criterion,
                      style=y_s,
                      device=device,
                      tensorboard_logger=log_name)

    trainer.train(data_loader, 4, checkpoint=ARGS.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args Style Transfer Training')
    parser.add_argument('--data_root', type=str, default='/content/data',
                        help='Root directory of the dataset')
    parser.add_argument('--style_folder', type=str, default='./mandostyles',
                        help='Directory of the style images')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='Learning rate for ADAM')
    parser.add_argument('-bs', '--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for the training')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of epochs for training')
    parser.add_argument('-cp', '--checkpoint', type=str, default=None,
                        help='Checkpoint to load in case of continued training')
    ARGS = parser.parse_args()
    main(ARGS)