from models import initialize_modules, getOptimizers
from tqdm import tqdm
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from utils.data import generate_dataset, combine_datasets
from torch.nn import functional as F
from utils.initialization import samples


if __name__=='__main__':
    
    train_env = [ generate_dataset(1.4, samples),  generate_dataset(1.5, samples)]
    test_env = [ generate_dataset(9.9, samples)]


    modules = initialize_modules(len(train_env))
    feature_extractor = modules[0]
    classifiers = modules[1:]

    optimizers = getOptimizers(modules)
    feature_extractor_optimizer = optimizers[0] 
    classifiers_optimizers = optimizers[1:]

    # Best Response Training of Invariant Risk minimization games
    with tqdm(range(10000)) as tepoch:
        for global_step in tepoch:
            
            for i, D in enumerate(train_env):
                Y_pred = torch.stack([ classifier(feature_extractor(D[0]))  for classifier in classifiers ]).mean(dim=0)
                loss = F.mse_loss(Y_pred, D[1])

                classifiers_optimizers[i].zero_grad()
                loss.backward()
                classifiers_optimizers[i].step()
            
            loss = 0
            for i, D in enumerate(train_env):
                Y_pred = torch.stack([ classifier(feature_extractor(D[0]))  for classifier in classifiers ]).mean(dim=0)
                loss += F.mse_loss(Y_pred, D[1])
            
            feature_extractor_optimizer.zero_grad()
            loss.backward()
            feature_extractor_optimizer.step()
            

            Y_pred = torch.stack([ classifier(feature_extractor(test_env[0][0]))  for classifier in classifiers ]).mean(dim=0)
            test_loss = F.mse_loss(test_env[0][1], Y_pred)
            tepoch.set_postfix_str(f'train:{loss}, test:{test_loss}')

            if global_step%50==0:
                w = torch.mean(torch.stack([ classifier.lin1.weight for classifier in classifiers ]), dim=0)
                print(torch.matmul(w, feature_extractor.lin1.weight))
