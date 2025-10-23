from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from torch.utils.data import Dataset
import struct
import numpy as np


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))

        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)  # [N, 28, 28]
        return data


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))

        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


class MnistDataset(Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.images = None
        self.labels = None
        self.categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        if train:
            self.images = load_mnist_images(Path(self.root).joinpath('train-images.idx3-ubyte'))
            self.labels = load_mnist_labels(Path(self.root).joinpath('train-labels.idx1-ubyte'))
        else:
            self.images = load_mnist_images(Path(self.root).joinpath('t10k-images.idx3-ubyte'))
            self.labels = load_mnist_labels(Path(self.root).joinpath('t10k-labels.idx1-ubyte'))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transforms:
            return self.transforms(image), label
        return image.reshape(1, 28, 28), label



