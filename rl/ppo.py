"""Asymmetric-actor-critic PPO for the vectorized drone environment.

The actor is trained on onboard observations; the critic on privileged ones.
Timeouts are treated as truncations: the value of the final state is bootstrapped
so the drone is not punished for an artificial episode limit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from rl.networks import Actor, Critic


@dataclass
class PPOConfig:
    rollout_steps: int = 64  # per environment per update
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    value_coef: float = 1.0
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    target_kl: float = 0.03  # stop the epoch loop early if the policy moves too far
    hidden: tuple[int, ...] = (256, 256)
    init_std: float = 0.5
    min_std: float = 0.1


class PPO:
    def __init__(
        self, actor_dim: int, critic_dim: int, cfg: PPOConfig | None = None, device: str = "cpu"
    ) -> None:
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device)
        self.actor = Actor(
            actor_dim, hidden=self.cfg.hidden, init_std=self.cfg.init_std, min_std=self.cfg.min_std
        ).to(self.device)
        self.critic = Critic(critic_dim, hidden=self.cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.cfg.lr
        )

    # ------------------------------------------------------------------ helpers
    def _t(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, obs_a: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        dist = self.actor.distribution(self._t(obs_a))
        a = dist.mean if deterministic else dist.sample()
        return a.cpu().numpy(), dist.log_prob(a).sum(-1).cpu().numpy()

    @torch.no_grad()
    def value(self, obs_c: np.ndarray) -> np.ndarray:
        return self.critic(self._t(obs_c)).cpu().numpy()

    # ------------------------------------------------------------------ rollout
    def collect(self, env, obs_a: np.ndarray, obs_c: np.ndarray):
        """Run rollout_steps in env. Returns (batch dict, last obs, list of finished episodes)."""
        cfg = self.cfg
        n, t_len = env.n, cfg.rollout_steps
        buf = {
            "obs_a": np.zeros((t_len, n, obs_a.shape[1]), np.float32),
            "obs_c": np.zeros((t_len, n, obs_c.shape[1]), np.float32),
            "act": np.zeros((t_len, n, 4), np.float32),
            "logp": np.zeros((t_len, n), np.float32),
            "rew": np.zeros((t_len, n), np.float32),
            "done": np.zeros((t_len, n), np.float32),
            "val": np.zeros((t_len, n), np.float32),
        }
        finished: list[dict] = []
        for t in range(t_len):
            a, logp = self.act(obs_a)
            v = self.value(obs_c)
            next_a, next_c, r, done, trunc, info = env.step(a)
            if trunc.any():  # bootstrap through timeouts
                r = r + cfg.gamma * trunc * self.value(info["terminal_critic_obs"])
            buf["obs_a"][t], buf["obs_c"][t] = obs_a, obs_c
            buf["act"][t], buf["logp"][t], buf["val"][t] = a, logp, v
            buf["rew"][t], buf["done"][t] = r, done
            for i in np.flatnonzero(done):
                finished.append(
                    {
                        "return": float(info["episode_return"][i]),
                        "length": int(info["episode_length"][i]),
                        "success": bool(info["success"][i]),
                        "collision": bool(info["collision"][i]),
                        "timeout": bool(info["timeout"][i]),
                        "stage": int(info["stage_id"][i]),
                        "waypoint_frac": float(
                            info["waypoints_reached"][i] / max(info["n_waypoints"][i], 1)
                        ),
                    }
                )
            obs_a, obs_c = next_a, next_c

        # generalized advantage estimation
        last_v = self.value(obs_c)
        adv = np.zeros((t_len, n), np.float32)
        gae = np.zeros(n, np.float32)
        for t in reversed(range(t_len)):
            next_v = last_v if t == t_len - 1 else buf["val"][t + 1]
            nonterm = 1.0 - buf["done"][t]
            delta = buf["rew"][t] + cfg.gamma * next_v * nonterm - buf["val"][t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
            adv[t] = gae
        buf["adv"] = adv
        buf["ret"] = adv + buf["val"]
        flat = {k: v.reshape(t_len * n, *v.shape[2:]) for k, v in buf.items()}
        return flat, obs_a, obs_c, finished

    # ------------------------------------------------------------------ update
    def update(self, batch: dict) -> dict:
        cfg = self.cfg
        obs_a, obs_c = self._t(batch["obs_a"]), self._t(batch["obs_c"])
        self.actor.norm.update(obs_a)
        self.critic.norm.update(obs_c)
        act, old_logp = self._t(batch["act"]), self._t(batch["logp"])
        adv, ret = self._t(batch["adv"]), self._t(batch["ret"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        size = obs_a.shape[0]
        mb = size // cfg.minibatches
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0, "clip_frac": 0.0}
        updates = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(size, device=self.device)
            epoch_kl = []
            for start in range(0, size - mb + 1, mb):
                idx = perm[start : start + mb]
                dist = self.actor.distribution(obs_a[idx])
                logp = dist.log_prob(act[idx]).sum(-1)
                ratio = torch.exp(logp - old_logp[idx])
                p1 = ratio * adv[idx]
                p2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[idx]
                policy_loss = -torch.min(p1, p2).mean()
                value_loss = (self.critic(obs_c[idx]) - ret[idx]).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self.opt.step()

                with torch.no_grad():
                    kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["kl"] += kl
                epoch_kl.append(kl)
                updates += 1
            if np.mean(epoch_kl) > cfg.target_kl:
                break
        out = {k: v / max(updates, 1) for k, v in stats.items()}
        out["action_std"] = float(
            torch.exp(torch.clamp(self.actor.log_std, self.actor.min_log_std, 1.0)).mean().item()
        )
        return out

    # ------------------------------------------------------------------ checkpoints
    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt": self.opt.state_dict(),
            "ppo_cfg": asdict(self.cfg),
        }

    def load_state_dict(self, sd: dict, load_optimizer: bool = True) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        if load_optimizer and "opt" in sd:
            self.opt.load_state_dict(sd["opt"])