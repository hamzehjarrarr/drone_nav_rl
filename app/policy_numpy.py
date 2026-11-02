"""Run the exported actor network with NumPy only (used by the app and evaluation).

The .npz file is written by rl/export.py and holds plain arrays: observation
normalization statistics and the weights of each linear layer. It is loaded with
allow_pickle=False, so it cannot execute code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NumpyPolicy:
    def __init__(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.mean = data["obs_mean"]
        self.std = np.sqrt(data["obs_var"] + 1e-8)
        self.clip = float(data["obs_clip"])
        self.n_layers = int(data["n_layers"])
        self.weights = [data[f"w{i}"] for i in range(self.n_layers)]
        self.biases = [data[f"b{i}"] for i in range(self.n_layers)]
        self.meta = json.loads(str(data["meta"])) if "meta" in data else {}
        self.profile = str(self.meta.get("profile", "kingfisher"))  # drone it was trained for
        self.obs_dim = self.mean.shape[0]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action (the Gaussian mean), clipped to [-1, 1]. obs: (n, obs_dim)."""
        x = np.clip((obs - self.mean) / self.std, -self.clip, self.clip)
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = x @ w.T + b
            if i < self.n_layers - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))  # ELU
        return np.clip(x, -1.0, 1.0)