import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.act = nn.SiLU(inplace=True)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):

        identity = self.skip(x)

        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))

        return x + identity


class AttentionBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.norm = nn.GroupNorm(8, channels)

        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)

        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):

        b, c, h, w = x.shape

        residual = x

        x = self.norm(x)
        x = x.view(b, c, h*w)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = torch.softmax(
            torch.bmm(q.transpose(1,2), k) / (c ** 0.5),
            dim=-1
        )

        out = torch.bmm(v, attn.transpose(1,2))

        out = self.proj(out)
        out = out.view(b, c, h, w)

        return residual + out


class Encoder(nn.Module):

    def __init__(self, latent_dim=256):
        super().__init__()

        ch = 64

        self.layers = nn.Sequential(

            nn.Conv2d(3, ch, 3, padding=1),

            ResBlock(ch, ch),
            nn.Conv2d(ch, ch*2, 4, stride=2, padding=1),

            ResBlock(ch*2, ch*2),
            nn.Conv2d(ch*2, ch*4, 4, stride=2, padding=1),

            ResBlock(ch*4, ch*4),
            nn.Conv2d(ch*4, ch*4, 4, stride=2, padding=1),

            ResBlock(ch*4, ch*6),
            nn.Conv2d(ch*6, ch*6, 4, stride=2, padding=1),

            ResBlock(ch*6, ch*6),
            nn.Conv2d(ch*6, ch*6, 4, stride=2, padding=1),

            #AttentionBlock(ch*6),

            ResBlock(ch*6, ch*6),
        )

        self.to_latent = nn.Conv2d(ch*6, latent_dim, 1)

    def forward(self, x):

        x = self.layers(x)
        return self.to_latent(x)


class UpBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.res = ResBlock(out_ch, out_ch)

    def forward(self, x):

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)

        return self.res(x)


class Decoder(nn.Module):

    def __init__(self, latent_dim=256):
        super().__init__()

        ch = 64

        self.from_latent = nn.Conv2d(latent_dim, ch*6, 1)

        self.mid = nn.Sequential(
            ResBlock(ch*6, ch*6),
            #AttentionBlock(ch*6),
            ResBlock(ch*6, ch*6),
        )

        self.up = nn.Sequential(

            UpBlock(ch*6, ch*6),
            UpBlock(ch*6, ch*4),
            UpBlock(ch*4, ch*4),
            UpBlock(ch*4, ch*2),
            UpBlock(ch*2, ch),
        )

        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.from_latent(x)
        x = self.mid(x)
        x = self.up(x)

        return self.out(x)


class Autoencoder(nn.Module):

    def __init__(self, latent_dim=256):
        super().__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):

        z = self.encoder(x)
        recon = self.decoder(z)

        return recon, z
    