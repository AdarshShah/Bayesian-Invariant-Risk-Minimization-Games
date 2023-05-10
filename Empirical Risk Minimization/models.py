from torch import nn
import torch
from torch.optim import Adam
from utils.initialization import device


class FeatureExtractor(nn.Module):

    def __init__(self) -> None:
        super(FeatureExtractor, self).__init__()
        self.lin1 = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.lin1(x)


class Classifier(nn.Module):

    def __init__(self) -> None:
        super(Classifier, self).__init__()
        self.lin1 = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.lin1(x)