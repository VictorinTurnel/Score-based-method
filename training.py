import torch
from tqdm import tqdm
from utils.sample_generator import sample_generator
from models.mlp import ScoreMLP
from losses.sliced_score_loss import sliced_score_matching_loss

import numpy as np
import matplotlib.pyplot as plt


training_iter = 10000
batch_size = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ScoreMLP().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in tqdm(range(training_iter)):

    x = sample_generator(batch_size)
    optimizer.zero_grad()
    loss = sliced_score_matching_loss(model, x)
    loss.backward()
    optimizer.step()

    if i%1000 == 0:
        print(f"Iter = {i} | Loss = {loss}")

torch.save(model, "./trained_models/score_mlp.pth")

