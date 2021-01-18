import torch
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import ImageTransformUNet
from VGG import MyVgg


def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512


loader = transforms.Compose(
    [
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

y_c = load_image("annahathaway.png")
y_s = load_image("style.jpg")

VGG = MyVgg().to(device).eval()
loss_mse = torch.nn.MSELoss()

image_transformer = ImageTransformUNet().to(device)
image_transformer.train()

# Hyperparameters
total_steps = 100
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam(image_transformer.parameters(), lr=learning_rate)

for step in range(total_steps):

    y_c_features = VGG(y_c)
    y_s_features = VGG(y_s)

    y_hat = image_transformer(y_c)
    y_hat_features = VGG(y_hat)

    # Loss is 0 initially
    style_loss = original_loss = 0

    # iterate through all the features for the chosen layers
    for gen_feature, orig_feature, style_feature in zip(
            y_hat_features, y_c_features, y_s_features
    ):

        # batch_size will just be 1
        batch_size, channel, height, width = gen_feature.shape

        #original_loss += torch.mean((gen_feature - orig_feature) ** 2)
        original_loss += loss_mse(gen_feature, orig_feature)

        # Compute Gram Matrix of generated
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )
        # Compute Gram Matrix of Style
        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )
        #style_loss += torch.mean((G - A) ** 2)
        style_loss += loss_mse(G, A)

    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        image_transformer.eval()
        generated = image_transformer(y_c)
        save_image(generated, "generated.png")
        image_transformer.train()