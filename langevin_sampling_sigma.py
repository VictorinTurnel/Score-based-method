import torch

model_path = "./trained_models/score_mlp_annealed.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(model_path).to(device)

def annealed_langevin_sampling(model, sigmas, epsilon = 0.1, T=1000, x_init = None, device = "cpu"):
    
    if x_init is None:
        x = torch.randn((1,2)).to(device) * 5
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