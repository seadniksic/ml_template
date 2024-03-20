import os
import torch
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class Dataset(Dataset):

    def __init__(self, image_path, label_path, transform=None, label_transform=None):
        self.image_path = image_path

        with open(label_path, "rb") as f:
          self.labels = list(pickle.load(f).items())

        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, idx):

        image = np.array(Image.open(os.path.join(self.image_path, self.labels[idx][0])), dtype=np.float32)[:,:,:3]

        label = self.labels[idx][1]

        # output to transform is a numpy array (float 32) of shape (W, H, C)
        if self.transform:
            image=self.transform(image)
        
        if self.label_transform:
            label = self.label_transform(label)

        return image, label

    def __len__(self):
        return len(self.labels)
