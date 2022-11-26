import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(128, 10)

    def forward(self,x):
        x = self.linear(x)
        return x