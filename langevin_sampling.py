import torch
import numpy as np
import matplotlib.pyplot as plt
from models.mlp import ScoreMLP
from tqdm import tqdm

nb_points = 100
model_path = "./trained_models/ScoreMLP.pth" #"./trained_models/ScoreMLP.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ScoreMLP()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

def langevin_sampling(model, T=1000, epsilon = 0.5, x_init = None, device = "cpu"):

    if x_init is None:
        x = torch.randn(1,2).to(device) * 1
    else:
        x = x_init.clone().to(device)

    x.requires_grad_(False)
    for t in range(T):
        s = model(x)
        z = torch.randn_like(x)
        x = x + 0.5 * epsilon * s + epsilon**0.5 * z

    return x.detach().cpu().numpy()

sampled_points = []
for i in tqdm(range(nb_points)):
    sampled_points.append(langevin_sampling(model, device=device)[0])

sampled_points = np.asarray(sampled_points)
plt.scatter(sampled_points[:,0],sampled_points[:,1])
plt.show()


