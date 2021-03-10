import glob
import os

import torch
from torch.utils.data import Dataset

class TensorDataset(Dataset):
    def __init__(self, images_path):
        self.images = glob.glob(os.path.join(images_path, '*.pt'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx < 0 or idx >len(self.images):
            raise IndexError("Index out of bound")

        image_path = self.images[idx]
        img = torch.load(image_path)

        return img, img