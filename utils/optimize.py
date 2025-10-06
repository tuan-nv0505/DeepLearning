from pathlib import Path
import sys

from sympy.physics.units import velocity

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch

class SGD:
    def __init__(self, params, lr=1e-3, momentum=0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(param) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            self.velocities[i].mul_(self.momentum).add_(param.grad * self.lr)
            with torch.no_grad():
                param.sub_(self.velocities[i])

    def zero_grad(self):
        for param in self.params:
            if param.grad is None:
                continue
            param.grad.zero_()

