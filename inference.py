import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from Unet import UNet, UNetSmall
from ITN import ImageTransformNet

from utils import load_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
])

def style_transfer():
  
    y_c = load_image("annahathaway.png", transformer).to(device)

    # load style model
    style_model = UNetSmall(8).to(device)
    style_model.load_state_dict(torch.load('UNetSmall_0309-15.model'))

    # process input image
    stylized = style_model(y_c, torch.tensor([1]).to(device)).cpu()
    save_image(stylized.data[0], 'gen.jpg')

if __name__ == "__main__":
  style_transfer()