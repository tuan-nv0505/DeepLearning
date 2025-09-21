from pathlib import Path
import sys
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

import torch

def softmax(x, dim=0):
    x_exp = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def cross_entropy_loss(logits, labels, dim=0):
    probs = softmax(logits, dim=dim)
    loss_sample = -torch.sum(torch.log(probs) * labels, dim=dim)
    return torch.mean(loss_sample)

class NeuralNetwork:
    def __init__(self,):
        self.weights_ = None

if __name__ == '__main__':
    weight = torch.tensor([
        [0.2, -0.5, 0.1, 2.0],
        [1.5, 1.3, 2.1, 0.0],
        [0, 0.25, 0.2, -0.3]
    ]).float()
    x = torch.tensor([0.5, 0.7, 1, 2]).float()
    labels = torch.tensor([1, 0, 0]).float()

    logits = torch.matmul(weight, x)
    ce_1 = cross_entropy_loss(logits, labels)
    print(ce_1)

    probs = softmax(logits)
    grad = (probs - labels).view(-1, 1) * x.view(1, -1)
    print((probs - labels).view(-1, 1))
    print(x.view(1, -1))
    weight = weight - 1 * grad
    logits = torch.matmul(weight, x)
    ce_2 = cross_entropy_loss(logits, labels)
    print(ce_2)