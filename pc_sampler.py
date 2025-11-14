import torch

def pc_sampler(model, sigmas, corrector_step, alpha=0.005, x_init=None, device="cuda"):
    
    if x_init is None:
        x = torch.randn((1,2)).to(device) * max(sigmas)
    else:
        x = x_init.clone().to(device)

    for i in range(len(sigmas) - 1):
        sigma_curr = sigmas[i]
        sigma_next = sigmas[i+1]

        s = model(x, )
        diff_sigma = sigma_curr**2 - sigma_next**2
        z = torch.randn_like(x)

        diffusion_std = torch.sqrt(diff_sigma)
        if sigma_next == sigmas[-1]:
            diffusion_std = 0
        x = x + s * diff_sigma + diffusion_std * z

        for _ in range(corrector_step):
            z = torch.randn_like(x)
            s = model(x, )
            x = x + alpha * s +(2 * alpha)**0.5 * z 



    return x

    