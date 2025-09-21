from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict
from matplotlib import pyplot as plt


class SimpleNN:
    def __init__(self, input_size, num_classes, learning_rate=0.01):
        self.weights_ = torch.randn(num_classes, input_size)
        self.biases_ = torch.ones(num_classes)
        self.learning_rate = learning_rate

    def softmax(self, logits):
        logits_exp = torch.exp(logits - torch.max(logits, dim=1, keepdim=True).values)
        return logits_exp / torch.sum(logits_exp, dim=1, keepdim=True)

    def forward(self, x):
        return x @ self.weights_.T + self.biases_

    def cross_entropy_loss(self, logits, one_hot_vector):
        probs = self.softmax(logits)
        loss_sample = -torch.sum(one_hot_vector * torch.log(probs + 1e-12), dim=1)
        return torch.mean(loss_sample)

    def backward(self, x, one_hot_vector):
        logits = self.forward(x)
        probs = self.softmax(logits)

        gradient = (probs - one_hot_vector).T @ x
        self.weights_ -= self.learning_rate * gradient

        gradient_bias = torch.sum(probs - one_hot_vector, dim=0)
        self.biases_ -= self.learning_rate * gradient_bias

        return self.cross_entropy_loss(logits, one_hot_vector)

def read_data(folder_path):
    x, y, labels, group = [], [], [], defaultdict(int)
    num_classes = len(list(Path(folder_path).iterdir()))
    for i, file in enumerate(Path(folder_path).iterdir()):
        if file.is_file() and file.name.endswith('.npy'):
            load_file = np.load(file)[:2000]
            x.append(load_file)
            one_hot_vector = np.zeros((1, num_classes))
            one_hot_vector[0][i] = 1
            y.extend([one_hot_vector for _ in range(load_file.shape[0])])
            labels.extend([file.name] * load_file.shape[0])
            group[file.name] = i
    return (
        torch.tensor(np.concatenate(x, axis=0),
        dtype=torch.float32),
        torch.tensor(np.concatenate(y, axis=0),
        dtype=torch.float32), labels, group
    )

if __name__ == '__main__':
    x, y, labels, category = read_data(PROJECT_DIR / 'datasets/quick_draw')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(x_train.numpy())
    x_train = scaler.transform(x_train.numpy())
    x_train = torch.tensor(x_train, dtype=torch.float32)

    x_test = scaler.transform(x_test.numpy())
    x_test = torch.tensor(x_test, dtype=torch.float32)

    model = SimpleNN(input_size=x_train.shape[1], num_classes=len(category), learning_rate=0.01)

    epochs = 50
    batch_size = 64

    loss_history = []
    train_acc_history = []
    test_acc_history = []

    for epoch in range(epochs):
        perm = torch.randperm(x_train.shape[0])
        x_train = x_train[perm]
        y_train = y_train[perm]

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, x_train.shape[0], batch_size):
            X_batch = x_train[i:i + batch_size]
            Y_batch = y_train[i:i + batch_size]

            logits = model.forward(X_batch)
            loss = model.cross_entropy_loss(logits, Y_batch)

            model.backward(X_batch, Y_batch)

            epoch_loss += loss.item()
            num_batches += 1

        loss_history.append(epoch_loss / num_batches)

        with torch.no_grad():
            # train accuracy
            train_logits = model.forward(x_train)
            train_probs = model.softmax(train_logits)
            train_preds = torch.argmax(train_probs, dim=1)
            train_true = torch.argmax(y_train, dim=1)
            train_acc = (train_preds == train_true).float().mean().item()
            train_acc_history.append(train_acc)

            # test accuracy
            test_logits = model.forward(x_test)
            test_probs = model.softmax(test_logits)
            test_preds = torch.argmax(test_probs, dim=1)
            test_true = torch.argmax(y_test, dim=1)
            test_acc = (test_preds == test_true).float().mean().item()
            test_acc_history.append(test_acc)

        print("Epoch {}/{} - Loss: {:.4f} - Train Acc: {:.4f} - Test Acc: {:.4f}".format(
            epoch + 1, epochs, epoch_loss / num_batches, train_acc, test_acc
        ))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_acc_history, label="Train Accuracy")
    plt.plot(range(1, epochs + 1), test_acc_history, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid()

    plt.show()


