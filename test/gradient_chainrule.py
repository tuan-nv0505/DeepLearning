from math import exp
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch


if __name__ == '__main__':
    x = torch.tensor(2.0, requires_grad=True)

    y = (x - 3)**2
    y.backward()
    print(x.grad)

    x.grad.zero_()
    y = (x - 4)**2
    y.backward()
    print(x.grad)
