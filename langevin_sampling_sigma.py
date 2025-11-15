import torch
import numpy as np
import matplotlib.pyplot as plt
from models.mlp import ScoreMLP_sigma
from tqdm import tqdm

nb_points = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
sigma_values = torch.exp(torch.linspace(torch.log(torch.tensor(20.0)),
                                        torch.log(torch.tensor(1.0)),
                                        10)).to(device)
model_path = "./trained_models/ScoreMLP_circle_sigma.pth" #"./trained_models/ScoreMLP_sigma.pth"
model = ScoreMLP_sigma()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

def annealed_langevin_sampling(model, sigmas, epsilon = 0.1, T=1000, x_init = None, device = "cpu"):
    
    if x_init is None:
        x = torch.randn((1,2)).to(device) * 3
    else:
        x = x_init.clone().to(device)

    x.requires_grad_(False)
    L = len(sigmas)
    for i in range(L):

        alpha = epsilon*sigmas[i]**2/sigmas[-1]**2
        for t in range(T):
            z = torch.randn_like(x)
            x = x + 0.5 * alpha * model(x,sigmas[i]) + alpha**0.5 * z
        
    return x.detach().cpu().numpy()

sampled_points = []
for i in tqdm(range(nb_points)):
    sampled_points.append(annealed_langevin_sampling(model, sigma_values, device=device)[0])

print(sampled_points)
sampled_points = np.asarray(sampled_points)
plt.scatter(sampled_points[:,0],sampled_points[:,1])
plt.show()