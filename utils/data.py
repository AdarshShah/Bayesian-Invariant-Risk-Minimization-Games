# Prepare Slab dataset

import torch
from utils.initialization import device

def generate_dataset(rho_e, size):
    rho = 1.0
    X_1 = rho*torch.randn(size, 2).to(device) # N(0, rho^2 I)
    Y = torch.matmul(X_1,torch.ones(2,1).to(device)) + rho*torch.randn(size, 1).to(device) # 1.T@X_1 + N(0, rho^2 I)
    X_2 = Y*torch.ones_like(X_1) + rho_e*rho*torch.randn(size, 2).to(device) # Y . 1 + N(0, (rho_e * rho)^2 I) 
    X = torch.concat((X_1,X_2), dim=1)
    return X, Y

def combine_datasets(datasets):
    X = torch.concat([dataset[0] for dataset in datasets])
    Y = torch.concat([dataset[1] for dataset in datasets])
    return X, Y

if __name__=='__main__':
    d_1 = generate_dataset(0.5, 4)
    d_2 = generate_dataset(1.0, 4)
    d_3 = generate_dataset(9.9, 4)
    print(d_1)
    print(d_2)
    print(d_3)
    print(combine_datasets([d_1, d_2, d_3]))