import torch
import numpy as np
import matplotlib.pyplot as plt
from models.unet import ScoreUNet

def langevin_sampling_images(model, batch_size, sigmas, T=1000, epsilon=1e-5, device="cuda"):

    with torch.no_grad():

        x = torch.randn(batch_size, 1, 28, 28).to(device)

        for sigma in sigmas:

            alpha = epsilon * (sigma**2) / (sigmas[-1]**2)
            sigma_batch = torch.ones(batch_size, device=device) * sigma

            for _ in range(T):
                z = torch.randn_like(x)
                score = model(x, sigma_batch)

                x = x + 0.5 * alpha * score + (alpha**0.5) * z

        x = (x * 0.5) + 0.5
        x = x.clamp(0, 1)

        return x.detach().cpu().numpy()

device = "cuda" if torch.cuda.is_available() else "cpu"

path = "./trained_models/ScoreUNet_MNIST.pth"
model = ScoreUNet()
model.load_state_dict(torch.load(path, map_location=device))
model.to(device)
model.eval()

sigma_values = torch.exp(torch.linspace(
    torch.log(torch.tensor(10.0)),
    torch.log(torch.tensor(1)),
    10
)).to(device)

gen_image = langevin_sampling_images(model, 1, sigma_values, device=device)
plt.imshow(gen_image[0,0], cmap='gray')
plt.show()
print(gen_image)