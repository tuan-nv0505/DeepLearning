from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch


if __name__ == '__main__':
    x = torch.tensor([1.0], requires_grad=True)
    y = x**2 + 2*x -3
    z = y**2 + 2*y

    y.retain_grad()
    z.backward()
    print(x.grad)
    print(y.grad)