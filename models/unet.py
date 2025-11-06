import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreUNet(nn.Module):

    def __init__(self, in_channels = 1, base_channels = 64, sigma_embed_dim = 128):
        super().__init__()

        self.encoder1 = nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv2d(in_channels=base_channels, out_channels=2*base_channels, kernel_size=3, stride=2, padding=1)
        self.encoder3 = nn.Conv2d(in_channels=2*base_channels, out_channels=4*base_channels, kernel_size=3, stride=2, padding=1)

        self.bottleneck = nn.Conv2d(in_channels=4*base_channels, out_channels=4*base_channels, kernel_size=3, padding=1 )

        self.decoder1 = nn.ConvTranspose2d(in_channels=4*base_channels, out_channels=2*base_channels, kernel_size=4, stride=2, padding=1)
        self.decoder2 = nn.ConvTranspose2d(in_channels=2*base_channels, out_channels=base_channels, kernel_size=4, stride=2, padding=1)
        self.decoder3 = nn.ConvTranspose2d(in_channels=base_channels, out_channels=in_channels, kernel_size=3, padding=1)

        self.activation_SiLU = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, base_channels)
        self.norm2 = nn.GroupNorm(8, 2*base_channels)
        self.norm3 = nn.GroupNorm(8, 4*base_channels)

        self.sigma_to_h1 = nn.Linear(in_features=sigma_embed_dim, out_features=base_channels)
        self.sigma_to_h2 = nn.Linear(in_features=sigma_embed_dim, out_features=base_channels*2)
        self.sigma_to_h3 = nn.Linear(in_features=sigma_embed_dim, out_features=base_channels*4)

        self.sigma_embed = nn.Sequential(
            nn.Linear(1,sigma_embed_dim),
            nn.SiLU(),
            nn.Linear(sigma_embed_dim, sigma_embed_dim)
        )

    def forward(self, x, sigma):

        s_emb = self.sigma_embed(sigma.unsqueeze(-1))

        h1 = self.activation_SiLU(self.norm1(self.encoder1(x))) + self.sigma_to_h1(s_emb)[:,:,None,None]
        h2 = self.activation_SiLU(self.norm2(self.encoder2(h1))) + self.sigma_to_h2(s_emb)[:,:,None, None]
        h3 = self.activation_SiLU(self.norm3(self.encoder3(h2))) + self.sigma_to_h3(s_emb)[:,:,None,None]

        hb = self.activation_SiLU(self.bottleneck(h3)) + self.sigma_to_h3(s_emb)[:,:,None,None]

        d1 = self.activation_SiLU(self.decoder1(hb) + h2)
        d2 = self.activation_SiLU(self.decoder2(d1) + h1)
        out = self.decoder3(d2)

        return out
