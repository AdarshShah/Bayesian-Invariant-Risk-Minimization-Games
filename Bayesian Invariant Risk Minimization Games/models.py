import torch
from torch import nn
from torch.nn import functional as F
from utils.initialization import device
from utils.data import generate_dataset

class FeatureExtractor(nn.Module):

    def __init__(self) -> None:
        super(FeatureExtractor, self).__init__()
        self.lin = nn.Linear(4,2,bias=False)

    def forward(self, x):
        return self.lin(x)

class Classifier(nn.Module):

    def __init__(self, mean, std, epsilon) -> None:
        super(Classifier, self).__init__()
        self.w = mean + std*epsilon
    
    def forward(self, x):
        return torch.matmul(x, self.w.T)

class AutoEncoder(nn.Module):

    def __init__(self) -> None:
        super(AutoEncoder, self).__init__()
        self.m_u = torch.nn.parameter.Parameter(torch.rand(1,2))
        self.std = torch.nn.parameter.Parameter(torch.rand(1,2))
        torch.nn.init.uniform_(self.m_u, -1, 1)
        torch.nn.init.uniform_(self.std, 0, 1)
    
    def recon_loss(self, X, Y, classifier, f_e): #minimize this
        return F.mse_loss(Y, classifier(f_e(X)))
    
    def KL_loss(self):   #maximize this
        return torch.sum(1 + torch.log(torch.square(self.std)) - torch.square(self.m_u) - torch.square(self.std))/2

    def sample(self, epsilon=None):
        if epsilon is None:
            epsilon =  torch.randn_like(self.m_u).to(device)
        return Classifier(self.m_u, self.std, epsilon)
    
    def fit(self, X, Y, f_e, epochs):
        optim = torch.optim.Adam([self.m_u, self.std], betas=(0.5, 0.5))
        for _ in range(epochs):
            loss = self.recon_loss(X, Y, self.sample(), f_e) - self.KL_loss()
            optim.zero_grad()
            loss.backward()
            optim.step()