import torch
import numpy as np
import matplotlib.pyplot as plt
from models.mlp import ScoreMLP, ScoreMLP_sigma

model_path = "./trained_models/ScoreMLP.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ScoreMLP()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

xs = np.linspace(-10, 10, 100)
ys = np.linspace(-10, 10, 100)
XX, YY = np.meshgrid(xs, ys)
grid = np.stack([XX.ravel(), YY.ravel()], axis=1)  
grid_torch = torch.from_numpy(grid).float().to(device)

with torch.no_grad():
    s_pred = model(grid_torch).cpu().numpy()

def true_score(x, mu1 = np.array([-5, -5]), mu2 = np.array([5, 5])):

    def gaussian_pdf(x, mu):
        return np.exp(-0.5*np.sum((x - mu)**2,axis=1))/(2*np.pi)
    
    gaussian1 = gaussian_pdf(x, mu1)
    gaussian2 = gaussian_pdf(x, mu2)
    gaussian = 0.2 * gaussian1 + 0.8 * gaussian2

    grad1 = -(x - mu1) * gaussian1[:,None]
    grad2 = -(x - mu2) * gaussian2[:,None]
    grad = (0.2 * grad1 + 0.8 * grad2) / gaussian[:, None]

    return grad

s_true = true_score(grid)

plt.figure(figsize=(8, 8))
skip = 5  
plt.quiver(grid[::skip,0], grid[::skip,1],
           s_pred[::skip,0], s_pred[::skip,1],
           color='r', label='predicted')
plt.quiver(grid[::skip,0], grid[::skip,1],
           s_true[::skip,0], s_true[::skip,1],
           color='b', alpha=0.5, label='true')
plt.legend()
plt.title("Scores : Learned (red) vs Ground-Truth (blue)")
plt.show()
