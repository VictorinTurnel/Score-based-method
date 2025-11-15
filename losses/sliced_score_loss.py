import torch

def sliced_score_matching_loss(model, x, M = 1):
    x = x.detach().requires_grad_(True)
    loss = 0
    for i in range(M):
        v = torch.rand_like(x)
        v = v / torch.linalg.norm(v, dim=1, keepdim=True)

        s = model(x) 

        vs = (v * s).sum(dim = 1)

        grad_vs = torch.autograd.grad(vs.sum(),x,create_graph=True)[0]
        vHv = (grad_vs * v).sum(dim=1)
        square = 0.5 * (s**2).sum(dim=1)

        loss += (vHv + square).mean()

    return 1/M * loss

"""def annealed_sliced_loss(model, x_noise, sigmas, M=1):
    x_noise = x_noise.detach().requires_grad_(True)
    B = x_noise.shape[0]
    loss = 0

    for _ in range(M):
        v = torch.randn_like(x_noise)
        v = v / torch.norm(v, dim=1, keepdim=True)

        s = model(x_noise, sigmas) 
        vs = (v * s).sum(dim=1)

        grad_vs = torch.autograd.grad(
            vs.sum(), x_noise, create_graph=True
        )[0]

        vHv = (grad_vs * v).sum(dim=1)
        square = 0.5 * (s * s).sum(dim=1)

        loss += (vHv + square).mean()

    return loss / M"""


def annealed_sliced_loss(model, x, sigmas, M = 1):
    x = x.detach().requires_grad_(True)
    B = x.shape[0]

    loss = 0
    for i in range(M):
        v = torch.rand_like(x)
        v = v / torch.linalg.norm(v.view(B, -1), dim=1, keepdim=True).view(B, 1, 1, 1)

        s = model(x, sigmas)

        vs = (v * s).view(B, -1).sum(dim = 1)

        grad_vs = torch.autograd.grad(vs.sum(),x,create_graph=True)[0]
        vHv = (grad_vs * v).view(B, -1).sum(dim=1)
        square = 0.5 * (s.view(B, -1)**2).sum(dim=1)

        loss += (vHv + square).mean()

    return 1/M * loss