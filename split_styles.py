from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import pickle
# from torchvision.utils import save_image
import torchvision.models as models
from sklearn.cluster import KMeans
from tqdm import tqdm

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg19(pretrained=True).to(device)

## freeze the layers
for param in model.parameters():
   param.requires_grad = False

# fc6
model.classifier = model.classifier[0]

transformer = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor()
])

data_root = '../MandoConcept'

paths = list(Path(data_root).rglob("*.jpg")) + list(Path(data_root).rglob("*.png"))
path_image = {}
if Path("features.pickle").is_file():
  with open(r"features.pickle", "rb") as output_file:
    path_image = pickle.load(output_file)
else:
  for path in tqdm(paths):
    image = load_image(path, transformer).to(device)
    feature = model(image)
    path_image[path] = feature.squeeze(0).numpy()
  
  with open(r"features.pickle", "wb") as output_file:
    pickle.dump(path_image, output_file)

print("fitting")
kmeans = KMeans(n_clusters=16, n_jobs=-1, random_state=42)
x = np.array(list(path_image.values()))

kmeans.fit(x)
print("done")

groups = {}
for path, cluster in zip(path_image.keys(), kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = [path]
    else:
        groups[cluster].append(path)

for group, gpaths in groups.items():
  p = Path(f'{data_root}/{group}')
  p.mkdir(parents=True, exist_ok=True)
  for path in gpaths:
    copyfile(path, f'{data_root}/{group}/{Path(path).name}')


# images = torch.cat(images, dim=0)
# print(images.shape)
# avg = torch.mean(images, dim=0)
# save_image(avg, 'avg.jpg')
