"""Actor and privileged critic networks with built-in observation normalization."""

from __future__ import annotations

import math

import torch
from torch import nn


class RunningNorm(nn.Module):
    """Tracks the running mean/variance of inputs and normalizes them (clipped to ±clip)."""

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-4) -> None:
        super().__init__()
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        n = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        m2 = self.var * self.count + batch_var * n + delta.pow(2) * self.count * n / total
        self.var = m2 / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


def mlp(inp: int, hidden: tuple[int, ...], out: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = inp
    for h in hidden:
        lin = nn.Linear(last, h)
        nn.init.orthogonal_(lin.weight, math.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ELU()]
        last = h
    head = nn.Linear(last, out)
    nn.init.orthogonal_(head.weight, out_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Gaussian policy over the 4 normalized CTBR actions. Sees onboard observations only."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 4,
        hidden: tuple[int, ...] = (256, 256),
        init_std: float = 0.5,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.min_log_std = math.log(min_std)
        self.norm = RunningNorm(obs_dim)
        self.net = mlp(obs_dim, hidden, act_dim, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(init_std)))
        self.hidden = hidden

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(obs))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.forward(obs)
        # exploration noise never drops below min_std, so the policy keeps trying new moves
        # when a harder curriculum stage starts (the app uses the mean action, no noise)
        std = torch.exp(torch.clamp(self.log_std, self.min_log_std, 1.0)).expand_as(mean)
        return torch.distributions.Normal(mean, std)


class Critic(nn.Module):
    """Value function. Sees the actor's observations plus privileged ground truth."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.net = mlp(obs_dim, hidden, 1, out_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(obs)).squeeze(-1)