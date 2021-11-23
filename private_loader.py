from time import sleep
from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import random

class Private_Dataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform

        imgs = np.load('/Users/bhanu/acad/sem1/CSCE636_DL/project/pytorch-cifar/data/cifar-10-batches-py/private_test_images_v3.npy')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Create a one hot label
        label = torch.zeros(10)

        # Transform the image by converting to tensor and normalizing it
        if self.transform:
            image = self.transform(self.data[idx])
        return image, label

if __name__ == "__main__":
    pass