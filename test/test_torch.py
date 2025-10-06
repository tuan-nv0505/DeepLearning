from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch


if __name__ == '__main__':
    a = torch.tensor([1,2,3,4,5])
    x = torch.zeros_like(a)
    print(a)
    print(x)