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
from sklearn.metrics import classification_report


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

    if torch.cuda.is_available():
        model.cuda()

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
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                predictions = model(images)   # predictions shape 64x10
                indices = torch.argmax(predictions.cpu(), dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions, labels)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        print("Epoch {}".format(epoch+1))
        print(classification_report(all_labels, all_predictions))



