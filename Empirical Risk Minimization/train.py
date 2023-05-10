from models import FeatureExtractor, Classifier
from tqdm import tqdm
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from utils.data import generate_dataset, combine_datasets
from torch.nn import functional as F
from utils.initialization import device, samples


if __name__=='__main__':
    
    train_env = [ generate_dataset(1.4, samples),  generate_dataset(1.5, samples)]
    test_env = [ generate_dataset(9.9, samples)]


    feature_extractor = FeatureExtractor().to(device)
    classifier = Classifier().to(device)

    optimizer = torch.optim.Adam(list(feature_extractor.parameters())+ list(classifier.parameters()))

    # Best Response Training of Invariant Risk minimization games
    with tqdm(range(10000)) as tepoch:
        for global_step in tepoch:

            loss = 0
            for i, D in enumerate(train_env):
                loss += F.mse_loss(classifier(feature_extractor(D[0])), D[1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            test_loss = F.mse_loss(test_env[0][1], classifier(feature_extractor(test_env[0][0])))
            tepoch.set_postfix_str(f'train:{loss/len(train_env)}, test:{test_loss}')

            if global_step%50==0:
                print(torch.matmul(classifier.lin1.weight, feature_extractor.lin1.weight))
