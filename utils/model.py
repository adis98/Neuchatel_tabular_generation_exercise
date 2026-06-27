from typing import Callable, Union
import torch
import torch.nn as nn
import torch.optim
import math
import torch.nn.functional as F

ModuleType = Union[str, Callable[..., nn.Module]]


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class TimeStepEmbedding(nn.Module):
    """
    Layer that embeds diffusion timesteps.
    Args:
        - dim (int): the dimension of the output.
        - max_period (int): controls the minimum frequency of the embeddings.
        - n_layers (int): number of dense layers
        - fourier (bool): whether to use random fourier features as embeddings
    """
    def __init__(
            self,
            dim: int,
            max_period: int = 10000,
            n_layers: int = 2,
            fourier: bool = False,
            scale=16,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.n_layers = n_layers
        self.fourier = fourier

        mid = (dim + 1) // 2  # ceil(dim / 2)
        input_dim = mid * 2   # will be dim+1 when dim is odd

        if fourier:
            self.register_buffer("freqs", torch.randn(mid) * scale)

        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(input_dim, input_dim))
            layers.append(nn.SiLU())
        self.fc = nn.Sequential(*layers, nn.Linear(input_dim, dim))

    def forward(self, timesteps):
        mid = (self.dim + 1) // 2

        if not self.fourier:
            T = self.max_period
            fs = torch.exp(-math.log(T) / mid * torch.arange(mid, dtype=torch.float32))
            fs = fs.to(timesteps.device)
            args = timesteps[:, None].float() * fs[None]
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, mid*2]
        else:
            x = timesteps.ger((2 * torch.pi * self.freqs).to(timesteps.dtype))
            emb = torch.cat([x.cos(), x.sin()], dim=1)                   # [B, mid*2]

        return self.fc(emb)  # [B, dim]


class DiffusionMLP(nn.Module):
    """
    Simple MLP denoiser for tabular diffusion.
    Input: concatenation of continuous + one-hot categorical features.
    Output: same shape as input (predicts v / epsilon / x0).
    """

    def __init__(
            self,
            in_dim: int,
            n_layers: int = 5,
            n_units: int = 796,
            emb_dim: int = 256,
            act: str = "relu",
            num_y_classes: int = None,
    ):
        super().__init__()
        self.time_emb = TimeStepEmbedding(emb_dim)

        self.y_cond = num_y_classes is not None
        if self.y_cond:
            self.y_emb = nn.Embedding(num_y_classes, emb_dim)

        # project input features to emb_dim to add with time embedding
        self.proj = nn.Linear(in_dim, emb_dim)

        # MLP backbone
        in_dims = [emb_dim] + (n_layers - 1) * [n_units]
        out_dims = (n_layers - 1) * [n_units] + [n_units]
        layers = []
        for i, (ind, outd) in enumerate(zip(in_dims, out_dims)):
            layers.append(nn.Linear(ind, outd))
            layers.append(nn.ReLU() if act == "relu" else nn.SiLU())
        self.fc = nn.Sequential(*layers)

        # final projection back to input dimension
        self.out = nn.Linear(n_units, in_dim)

    def forward(self, x, time, c=None):
        # x:    [B, in_dim]
        # time: [B]  (values in [0, 1], will be scaled internally)

        cond = self.time_emb(time * 1000)  # [B, emb_dim]
        if self.y_cond and c is not None:
            cond = cond + F.silu(self.y_emb(c))  # [B, emb_dim]

        x = self.proj(x) + cond  # [B, emb_dim]
        x = self.fc(x)  # [B, n_units]
        return self.out(x)
