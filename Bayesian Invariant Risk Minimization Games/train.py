from utils.data import generate_dataset, combine_datasets
from models import FeatureExtractor, Classifier, AutoEncoder, torch
from utils.initialization import device, samples
from tqdm import tqdm

if __name__=='__main__':
    
    train_env = [ generate_dataset(1.4, samples),  generate_dataset(1.5, samples)]
    train_env.append(combine_datasets(train_env))
    test_env = [ generate_dataset(9.9, samples)]

    f_e = FeatureExtractor().to(device)
    q_u = [ AutoEncoder().to(device) for _ in train_env]

    lamda = 1
    with tqdm(range(10000)) as tepoch:
        for ep in tepoch:

                        
            # Update posteriors distribution of classifiers for each training environment
            for q, D in zip(q_u, train_env):
                q.fit(D[0], D[1], f_e, 20)
            
            # Update feature extractor only
            loss = 0
            optim = torch.optim.Adam(f_e.parameters(), betas=(0.5, 0.5))
            for _ in range(10):
                epsilon = torch.randn_like(q_u[-1].m_u).to(device)
                [ loss := loss + (1+lamda)*q_u[-1].recon_loss(D[0], D[1], q_u[-1].sample(epsilon), f_e)-lamda*q.recon_loss(D[0], D[1], q.sample(epsilon), f_e)  for q, D in zip(q_u[:-1], train_env[:-1]) ]
            loss = loss/10
            optim.zero_grad()
            loss.backward()
            optim.step()

            epsilon = torch.zeros_like(q_u[-1].m_u).to(device)
            test_acc = q_u[-1].recon_loss(test_env[0][0], test_env[0][1], q_u[-1].sample(epsilon), f_e)
            train_acc = q_u[-1].recon_loss(train_env[-1][0], train_env[-1][1], q_u[-1].sample(epsilon), f_e)
            tepoch.set_postfix_str(f'train:{train_acc}, test:{test_acc}')

            if ep%50==0:
                print(torch.matmul(q_u[-1].m_u, f_e.lin.weight))
