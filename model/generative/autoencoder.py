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

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):

        identity = self.skip(x)

        x = F.silu(self.norm1(x))
        x = self.conv1(x)

        x = F.silu(self.norm2(x))
        x = self.conv2(x)

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
        x_in = x

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

        return x_in + out
    


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(

            nn.Conv2d(3, 64, 3, padding=1),

            ResBlock(64, 64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),

            ResBlock(128, 128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),

            ResBlock(256, 256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),

            ResBlock(256, 512),
            nn.Conv2d(512, 512, 4, stride=2, padding=1),

            ResBlock(512, 512),
            nn.Conv2d(512, 512, 4, stride=2, padding=1),

            AttentionBlock(512),

            ResBlock(512, 512)
        )

    def forward(self, x):
        return self.layers(x)
    

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

    def __init__(self):
        super().__init__()

        self.mid = nn.Sequential(
            ResBlock(512, 512),
            AttentionBlock(512),
            ResBlock(512, 512),
        )

        self.up = nn.Sequential(

            UpBlock(512, 512),
            UpBlock(512, 256),
            UpBlock(256, 256),
            UpBlock(256, 128),
            UpBlock(128, 64),
        )

        self.out = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.mid(x)
        x = self.up(x)

        return self.out(x)
    

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):

        z = self.encoder(x)
        recon = self.decoder(z)

        return recon, z