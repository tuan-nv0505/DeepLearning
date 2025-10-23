import sys
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT_DIR))

from torch.utils.data import Dataset
import pickle
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image


class CIFAR10Dataset(Dataset):

    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.images = []
        self.labels = []
        self.categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        if self.train:
            path_files = [str(PROJECT_ROOT_DIR.joinpath(root, '{}'.format(x.name))) for x in Path(root).iterdir() if 'data_batch_' in x.name]
        else:
            path_files = [str(PROJECT_ROOT_DIR.joinpath(root, 'test_batch'))]
        for x in path_files:
            with open(x, 'rb') as file_open:
                data = pickle.load(file_open, encoding='bytes')
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = torch.from_numpy(self.images[index].reshape(3, 32, 32)).float() / 255
        return img, self.labels[index]
