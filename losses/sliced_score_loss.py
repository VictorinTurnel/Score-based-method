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