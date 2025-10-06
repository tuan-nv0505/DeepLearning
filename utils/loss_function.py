from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch


def softmax(output, dim=1):
    exp = torch.exp(output - torch.max(output, dim=dim, keepdim=True).values)
    return exp / torch.sum(exp, dim=dim,keepdim=True) 

def cross_entropy_loss(output, label, dim=1):
    probs = softmax(output)
    onehot_label = torch.zeros(output.shape)
    onehot_label.scatter_(1, label.unsqueeze(1), 1)
    return torch.mean(-torch.sum(torch.log(probs + 1e-9) * onehot_label, dim=dim))
    

