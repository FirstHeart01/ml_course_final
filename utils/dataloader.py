import gzip

import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image


def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as f:
        y_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as f:
        x_datas = np.frombuffer(f.read(), np.uint8, offset=16)
        x_datas = x_datas.reshape(len(y_labels), (28, 28))
    return x_datas, y_labels


class Mnistdataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.data, self.targets = torch.load(root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image.numpy(), mode='L')
        if self.transform is not None:
            image = self.transform(image)
            image = transforms.ToTensor()(image)
        sample = {'image': image, 'target': target}
        return sample
