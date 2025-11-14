import torch

def sample_generator(batch_size):

    mean1 = torch.tensor([-5, -5])
    mean2 = torch.tensor([5, 5])

    alpha = (torch.rand(batch_size) < 0.2).float().unsqueeze(1)

    x1 = torch.randn(batch_size, 2) + mean1
    x2 = torch.randn(batch_size, 2) + mean2

    return alpha * x1 + (1 - alpha) * x2


def sample_circle(batch_size, radius=5.0, noise_std=1):
    theta = 2 * torch.pi * torch.rand(batch_size)   # angle uniforme
    x = radius * torch.cos(theta) + noise_std * torch.randn(batch_size)
    y = radius * torch.sin(theta) + noise_std * torch.randn(batch_size)
    return torch.stack([x, y], dim=1)


