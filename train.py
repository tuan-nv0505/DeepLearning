from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from neural_network.neural_network import SimpleNeuralNetwork
from build_datasets.animals_dataset import AnimalsDataset
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader


if __name__ == '__main__':
    num_epochs = 100

    transforms = Compose([
        ToTensor(),
        Resize((200, 200))
    ]) 
    
    data_train = AnimalsDataset(root='datasets/animals', train=True, transforms=transforms)
    training_dataloader = DataLoader(
        dataset = data_train,
        batch_size = 32,
        shuffle = True,
        num_workers = 1,
        drop_last = True
    )

    data_test = AnimalsDataset(root='datasets/animals', train=False, transforms=transforms)
    test_dataloader = DataLoader(
        dataset = data_test,
        batch_size = 32,
        shuffle = False,
        num_workers = 1,
        drop_last = False
    )
    
    model = SimpleNeuralNetwork(num_classes=len(data_train.categories))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(training_dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            output = model(images)
            loss = criterion(output, labels)
            print('Epoch: [{}]/[{}] | Iteration: [{}]/[{}] | Loss: {:4f}'.format(epoch, num_epochs, i, len(training_dataloader), loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_labels = []
        all_predictions = []
        for images, labels in test_dataloader:
            all_labels.extend(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                output = model(images)
                loss = criterion(output, labels)
                indices = torch.argmax(output)
                all_predictions.extend(indices)

        print(all_labels)
        print('-------------')
        print(all_predictions)



