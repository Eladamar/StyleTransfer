import argparse

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from Unet import UNet
from utils import load_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def style_transfer(style_model, y_c, style, output):
    stylized = style_model(y_c, torch.tensor([style]).to(device))
    stylized = transforms.Resize(ARGS.output_size)(stylized).cpu()
    save_image(stylized.data[0], f'{ARGS.output}/gen{style}.jpg')

def main(ARGS):

    transformer = transforms.Compose([
        transforms.Resize(ARGS.input_size),
        transforms.CenterCrop(ARGS.input_size),
        transforms.ToTensor()
    ])
  
    y_c = load_image(ARGS.content_image, transformer).to(device)

    # load style model
    style_model = UNet(ARGS.styles_num).to(device)
    style_model.load_state_dict(torch.load('UNet_0316-15.model'))

    # process input image
    if ARGS.style != -1:
        style_transfer(style_model, ARGS.content_image, ARGS.style, ARGS.output)
        return

    for style in range(ARGS.styles_num):
        style_transfer(style_model, ARGS.content_image, style, ARGS.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Style An Image')
    parser.add_argument('--model', type=str, default='./model.model',
                        help='Path to model')
    parser.add_argument('--content_image', type=str, default='./content.jpg',
                        help='Path to content image')
    parser.add_argument('-sn', '--styles_num', type=int, default=16,
                        help='Number of Style images trained in model')
    parser.add_argument('--input_size', type=int, default=256,
                        help='Image size used in training')
    parser.add_argument('--output_size', type=int, default=256,
                        help='Stylized output size')
    parser.add_argument('--style', nargs='+', type=int,
                        help='Style indices to use on image')
    parser.add_argument('--output', type=str, default='./',
                        help='Path to output folder')
    ARGS = parser.parse_args()

    main(ARGS)