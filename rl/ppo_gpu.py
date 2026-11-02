"""PPO rollouts that stay on the GPU, for simgpu.TorchDroneEnv.

Same algorithm and update as rl.ppo.PPO; only data collection differs: observations,
actions, rewards and advantages never leave the device, and episode statistics are
copied to the CPU once per rollout instead of every step.
"""

from __future__ import annotations

import torch

from rl.ppo import PPO


class GPUPPO(PPO):
    @torch.no_grad()
    def collect(self, env, obs_a, obs_c):
        cfg = self.cfg
        if hasattr(env, "refresh_pools"):
            env.refresh_pools()
        n, t_len, dev = env.n, cfg.rollout_steps, self.device
        obs_a, obs_c = obs_a.to(dev, torch.float32), obs_c.to(dev, torch.float32)

        def buf(*shape):
            return torch.zeros(t_len, n, *shape, device=dev)

        b = {
            "obs_a": buf(obs_a.shape[1]),
            "obs_c": buf(obs_c.shape[1]),
            "act": buf(4),
            "logp": buf(),
            "rew": buf(),
            "done": buf(),
            "val": buf(),
        }
        ep_keys = ("success", "collision", "timeout")
        ep = {k: torch.zeros(t_len, n, dtype=torch.bool, device=dev) for k in ep_keys}
        ep_ret, ep_len, ep_stage, ep_wpf = buf(), buf(), buf(), buf()

        for t in range(t_len):
            dist = self.actor.distribution(obs_a)
            a = dist.sample()
            logp = dist.log_prob(a).sum(-1)
            v = self.critic(obs_c)
            next_a, next_c, r, done, trunc, info = env.step(a)
            # bootstrap through timeouts (computed for every env: no GPU -> CPU sync)
            term_v = self.critic(info["terminal_critic_obs"].to(torch.float32))
            r = r.to(torch.float32) + cfg.gamma * trunc.float() * term_v
            b["obs_a"][t], b["obs_c"][t] = obs_a, obs_c
            b["act"][t], b["logp"][t], b["val"][t] = a, logp, v
            b["rew"][t], b["done"][t] = r, done.float()
            for k in ep_keys:
                ep[k][t] = info[k] & done
            ep_ret[t] = info["episode_return"].float()
            ep_len[t] = info["episode_length"].float()
            ep_stage[t] = info["stage_id"].float()
            ep_wpf[t] = info["waypoints_reached"] / torch.clamp(info["n_waypoints"], min=1)
            obs_a, obs_c = next_a.to(torch.float32), next_c.to(torch.float32)

        last_v = self.critic(obs_c)
        adv = torch.zeros(t_len, n, device=dev)
        gae = torch.zeros(n, device=dev)
        for t in reversed(range(t_len)):
            next_v = last_v if t == t_len - 1 else b["val"][t + 1]
            nonterm = 1.0 - b["done"][t]
            delta = b["rew"][t] + cfg.gamma * next_v * nonterm - b["val"][t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
            adv[t] = gae
        b["adv"], b["ret"] = adv, adv + b["val"]
        flat = {k: v.reshape(t_len * n, *v.shape[2:]) for k, v in b.items()}

        # finished episodes -> plain Python dicts (one copy to the CPU per rollout)
        done_mask = b["done"].bool()
        rows = {
            "return": ep_ret[done_mask],
            "length": ep_len[done_mask],
            "stage": ep_stage[done_mask],
            "waypoint_frac": ep_wpf[done_mask],
            **{k: v[done_mask] for k, v in ep.items()},
        }
        rows = {k: v.cpu().tolist() for k, v in rows.items()}
        finished = [
            {
                "return": float(rows["return"][i]),
                "length": int(rows["length"][i]),
                "success": bool(rows["success"][i]),
                "collision": bool(rows["collision"][i]),
                "timeout": bool(rows["timeout"][i]),
                "stage": int(rows["stage"][i]),
                "waypoint_frac": float(rows["waypoint_frac"][i]),
            }
            for i in range(len(rows["return"]))
        ]
        return flat, obs_a, obs_c, finished