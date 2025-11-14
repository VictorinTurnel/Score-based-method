import torch
import numpy as np
import matplotlib.pyplot as plt
from models.mlp import ScoreMLP


model_path = "./trained_models/score_mlp_spirale.pth"

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

def true_score(x, radius=5.0, sigma=1):
    x = torch.from_numpy(x)
    norm = torch.norm(x, dim=1, keepdim=True)
    s = - (norm - radius) / (sigma**2) * (x / norm)
    return s


s_true = true_score(grid)

plt.figure(figsize=(8, 8))
skip = 3
plt.quiver(grid[::skip,0], grid[::skip,1],
           s_pred[::skip,0], s_pred[::skip,1],
           color='r', label='predicted')
plt.quiver(grid[::skip,0], grid[::skip,1],
           s_true[::skip,0], s_true[::skip,1],
           color='b', alpha=0.5, label='true')
plt.legend()
plt.title("Scores : Learned (red) vs Ground-Truth (blue)")
plt.show()
