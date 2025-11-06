import torch

def sample_generator(batch_size):

    mean1 = torch.tensor([-5, -5])
    mean2 = torch.tensor([5, 5])

    alpha = (torch.rand(batch_size) < 0.2).float().unsqueeze(1)

    x1 = torch.randn(batch_size, 2) + mean1
    x2 = torch.randn(batch_size, 2) + mean2

    return alpha * x1 + (1 - alpha) * x2

