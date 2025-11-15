import torch
import argparse
from tqdm import tqdm
from utils.sample_generator import sample_generator, sample_circle
from models.mlp import ScoreMLP, ScoreMLP_sigma
from losses.sliced_score_loss import sliced_score_matching_loss, annealed_sliced_loss
import matplotlib.pyplot as plt

def training(batch_size = 1024, training_iter = 10000, sigma = False, out_name = "score_mlp"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ScoreMLP_sigma() if sigma else ScoreMLP()
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sigma_values = torch.exp(torch.linspace(torch.log(torch.tensor(20.0)),
                                        torch.log(torch.tensor(1.0)),
                                        10)).to(device)


    for i in tqdm(range(training_iter)):

        x = sample_circle(batch_size,5,1).to(device) #sample_generator(batch_size).to(device)
        optimizer.zero_grad()
        if sigma:
            index = torch.randint(0, len(sigma_values), (x.shape[0],)).to(device)
            sigmas = sigma_values[index]
            noise = torch.randn_like(x) * sigmas.view(-1,1)
            x_noise = x + noise
            loss = annealed_sliced_loss(model, x_noise, sigmas) 
        else:
            loss = sliced_score_matching_loss(model, x)
        loss.backward()
        optimizer.step()

        if i%1000 == 0:
            print(f"Iter = {i} | Loss = {loss}")

        
    
    

    torch.save(model.state_dict(), f"./trained_models/{out_name}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--training_iter", type=int, default=10000)
    parser.add_argument("--out_name", type=str, default="score_mlp")
    parser.add_argument("--sigma", action="store_true")
    args = parser.parse_args()

    training(args.batch_size, args.training_iter, args.sigma, args.out_name)



