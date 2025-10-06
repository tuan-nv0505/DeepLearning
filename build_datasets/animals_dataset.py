from pathlib import Path
import sys

from numpy import mask_indices
from torchvision import transforms
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose


class AnimalsDataset(Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.categories = [category.name for category in Path(root).joinpath('train').iterdir() if category.is_dir()]
        self.path_files = []
        self.labels = []
        self.transforms = transforms
        
        for idx, category in enumerate(self.categories):
            path_category = ROOT_DIR.joinpath(self.root, '{}/{}'.format('train' if self.train else 'test', category))
            for path_file in path_category.iterdir():
                self.path_files.append(str(path_category.joinpath(path_file.name)))
                self.labels.append(idx)

    def __len__(self):
        return int(len(self.labels) * 1)

    def __getitem__(self, index):
        data = Image.open(self.path_files[index]).convert('RGB')
        if self.transforms is not None:
            return self.transforms(data), self.labels[index] #type:ignore
        return data, self.labels[index]

if __name__ == '__main__':
    transforms = Compose([
        Resize((200, 200)),
        ToTensor()
    ])
    animals_dataset = AnimalsDataset(root='datasets/animals', train=True, transforms=transforms)
    training_dataloader = DataLoader(
        dataset = animals_dataset,
        batch_size = 32,
        shuffle = True,
        num_workers = 0,  
        drop_last = True
    )

    for images, labels in training_dataloader:
        print(images.shape)
        print(labels)


