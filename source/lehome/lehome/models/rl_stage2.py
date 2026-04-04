"""
Stage 2 RL components: Normalizer, ReplayBuffer, Actor, Critic, Trainer.

Implements TD3+BC (Twin Delayed Deep Deterministic Policy Gradient with BC regularization)
for refining SmolVLA action chunks using RL Token state representation.

Paper: "RL Token: Bootstrapping Online RL with VLA Models" (Physical Intelligence, 2025)
"""

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ═══════════════════════════════════════════════════════════════════
# SimpleNormalizer — dataset stats only, no running stats
# ═══════════════════════════════════════════════════════════════════


class SimpleNormalizer:
    """Normalizer using dataset stats.json only. No warmup, no running stats.

    Stats source: Datasets/example/top_long_merged/meta/stats.json
    Contains: observation.state (mean, std), action (mean, std)
    """

    def __init__(self, stats_path: str, device: torch.device):
        with open(stats_path) as f:
            stats = json.load(f)
        self.pos_mean = torch.tensor(stats["observation.state"]["mean"], device=device)
        self.pos_std = torch.tensor(stats["observation.state"]["std"], device=device)
        self.act_mean = torch.tensor(stats["action"]["mean"], device=device)
        self.act_std = torch.tensor(stats["action"]["std"], device=device)

    def normalize_state(self, joint_pos: torch.Tensor) -> torch.Tensor:
        return (joint_pos - self.pos_mean) / self.pos_std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.act_mean) / self.act_std

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.act_std + self.act_mean

    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return state * self.pos_std + self.pos_mean


# ═══════════════════════════════════════════════════════════════════
# ReplayBuffer — circular buffer on GPU, chunk-level transitions
# ═══════════════════════════════════════════════════════════════════


class ReplayBuffer:
    """Circular replay buffer storing chunk-level transitions on GPU.

    Fields:
        z_rl:        (capacity, 960)       raw (no normalization)
        s_p:         (capacity, 12)        normalized joint_pos
        ref_action:  (capacity, C, 12)     VLA ã (normalized space)
        action:      (capacity, C, 12)     actor output (normalized space)
        reward:      (capacity,)           discounted chunk return
        next_z_rl:   (capacity, 960)       raw
        next_s_p:    (capacity, 12)        normalized
        done:        (capacity,)           bool
    """

    def __init__(
        self,
        capacity: int,
        z_rl_dim: int = 960,
        state_dim: int = 12,
        chunk_size: int = 10,
        action_dim: int = 12,
        device: torch.device = torch.device("cuda"),
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.z_rl = torch.zeros(capacity, z_rl_dim, device=device)
        self.s_p = torch.zeros(capacity, state_dim, device=device)
        self.ref_action = torch.zeros(capacity, chunk_size, action_dim, device=device)
        self.action = torch.zeros(capacity, chunk_size, action_dim, device=device)
        self.reward = torch.zeros(capacity, device=device)
        self.next_z_rl = torch.zeros(capacity, z_rl_dim, device=device)
        self.next_s_p = torch.zeros(capacity, state_dim, device=device)
        self.done = torch.zeros(capacity, dtype=torch.bool, device=device)

    def add(
        self,
        z_rl: torch.Tensor,
        s_p: torch.Tensor,
        ref_action: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_z_rl: torch.Tensor,
        next_s_p: torch.Tensor,
        done: bool,
    ):
        """Add a single chunk-level transition."""
        idx = self.ptr % self.capacity
        self.z_rl[idx] = z_rl.squeeze(0)
        self.s_p[idx] = s_p.squeeze(0)
        self.ref_action[idx] = ref_action.squeeze(0)
        self.action[idx] = action.squeeze(0)
        self.reward[idx] = reward
        self.next_z_rl[idx] = next_z_rl.squeeze(0)
        self.next_s_p[idx] = next_s_p.squeeze(0)
        self.done[idx] = done
        self.ptr += 1
        self.size = min(self.ptr, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "z_rl": self.z_rl[idx],
            "s_p": self.s_p[idx],
            "ref_action": self.ref_action[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_z_rl": self.next_z_rl[idx],
            "next_s_p": self.next_s_p[idx],
            "done": self.done[idx],
        }

    def __len__(self) -> int:
        return self.size


# ═══════════════════════════════════════════════════════════════════
# Helper: MLP builder
# ═══════════════════════════════════════════════════════════════════


def _make_mlp(input_dim: int, hidden_dim: int, num_layers: int, output_dim: int) -> nn.Module:
    layers = []
    in_d = input_dim
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(in_d, hidden_dim), nn.ReLU()])
        in_d = hidden_dim
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════════
# RLActor — Gaussian policy with reference conditioning
# ═══════════════════════════════════════════════════════════════════


class RLActor(nn.Module):
    """Actor network: [z_rl(960) + s_p(12)] + [ref(120)] → action_chunk(120).

    Uses fixed exploration noise σ = exp(-5) ≈ 0.0067.
    50% reference dropout during training.
    """

    def __init__(
        self,
        z_rl_dim: int = 960,
        state_dim: int = 12,
        chunk_size: int = 10,
        action_dim: int = 12,
        hidden_dim: int = 512,
        num_layers: int = 3,
        fixed_std: float = 0.0067,
        ref_dropout: float = 0.5,
    ):
        super().__init__()
        self.z_rl_dim = z_rl_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.fixed_std = fixed_std
        self.ref_dropout = ref_dropout

        x_dim = z_rl_dim + state_dim  # 972
        ref_dim = chunk_size * action_dim  # 120
        output_dim = chunk_size * action_dim  # 120

        self.net = _make_mlp(x_dim + ref_dim, hidden_dim, num_layers, output_dim)

    def apply_ref_dropout(self, ref_action: torch.Tensor, training: bool = True) -> torch.Tensor:
        """50% dropout: replace ref with zeros."""
        if training and self.ref_dropout > 0:
            mask = torch.rand(ref_action.shape[0], 1, device=ref_action.device) > self.ref_dropout
            ref_action = ref_action * mask.float().unsqueeze(-1)
        return ref_action

    def forward(self, z_rl: torch.Tensor, s_p: torch.Tensor, ref_action: torch.Tensor) -> torch.Tensor:
        """Training forward: reparameterization a = mu + σ·ε."""
        ref = self.apply_ref_dropout(ref_action.flatten(1), training=self.training)
        x = torch.cat([z_rl, s_p, ref], dim=-1)
        mu = self.net(x)

        if self.training:
            noise = torch.randn_like(mu) * self.fixed_std
            return mu + noise
        return mu

    def get_deterministic_action(
        self, z_rl: torch.Tensor, s_p: torch.Tensor, ref_action: torch.Tensor
    ) -> torch.Tensor:
        """Eval forward: return mu without noise, no dropout."""
        ref = ref_action.flatten(1)
        x = torch.cat([z_rl, s_p, ref], dim=-1)
        mu = self.net(x)
        return mu


# ═══════════════════════════════════════════════════════════════════
# TwinCritic — ensemble of 2 Q-functions
# ═══════════════════════════════════════════════════════════════════


class TwinCritic(nn.Module):
    """Critic network: [z_rl(960) + s_p(12)] + [action(120)] → Q1, Q2.

    Symmetric state input (same as actor).
    """

    def __init__(
        self,
        z_rl_dim: int = 960,
        state_dim: int = 12,
        chunk_size: int = 10,
        action_dim: int = 12,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        x_dim = z_rl_dim + state_dim  # 972
        action_flat_dim = chunk_size * action_dim  # 120

        self.q1 = _make_mlp(x_dim + action_flat_dim, hidden_dim, num_layers, output_dim=1)
        self.q2 = _make_mlp(x_dim + action_flat_dim, hidden_dim, num_layers, output_dim=1)

    def forward(self, z_rl: torch.Tensor, s_p: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_rl, s_p, action.flatten(1)], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, z_rl: torch.Tensor, s_p: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_rl, s_p, action.flatten(1)], dim=-1)
        return self.q1(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# RLTTrainer — TD3+BC training orchestrator
# ═══════════════════════════════════════════════════════════════════


class RLTTrainer:
    """TD3+BC trainer with:
    - Twin critics with clipped double-Q
    - Target policy smoothing (TD3 §4.3)
    - Delayed actor updates (every 2 critic steps)
    - BC regularization: L_π = -Q1(x,a) + β * ||a - ã||²
    - Reference action dropout (50%)
    - Target networks with Polyak averaging (τ=0.005)
    """

    def __init__(
        self,
        actor: RLActor,
        critic: TwinCritic,
        device: torch.device,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        beta: float = 0.1,
        target_noise_std: float = 0.2,
        noise_clip: float = 0.5,
        chunk_size: int = 10,
        actor_delay: int = 2,
        grad_clip: float = 1.0,
    ):
        self.device = device
        self.gamma_chunk = gamma ** chunk_size  # 0.99^10 ≈ 0.904
        self.tau = tau
        self.beta = beta
        self.target_noise_std = target_noise_std
        self.noise_clip = noise_clip
        self.actor_delay = actor_delay
        self.grad_clip = grad_clip
        self.step_count = 0

        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_target = copy.deepcopy(actor).to(device)
        self.critic_target = copy.deepcopy(critic).to(device)

        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

    def update_critic(self, batch: dict) -> dict:
        with torch.no_grad():
            # Target actions with TD3 smoothing
            next_a = self.actor_target.get_deterministic_action(
                batch["next_z_rl"], batch["next_s_p"], batch["ref_action"]
            )
            noise = (torch.randn_like(next_a) * self.target_noise_std).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_a_smooth = (next_a + noise).clamp(-1, 1)

            # Clipped double-Q target
            q1_t, q2_t = self.critic_target(
                batch["next_z_rl"], batch["next_s_p"], next_a_smooth
            )
            q_target = torch.min(q1_t, q2_t)
            td_target = batch["reward"] + self.gamma_chunk * (~batch["done"]) * q_target

        q1, q2 = self.critic(batch["z_rl"], batch["s_p"], batch["action"])
        loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()

        return {"critic_loss": loss.item(), "q1_mean": q1.mean().item()}

    def update_actor(self, batch: dict) -> dict:
        # Actor forward (with ref dropout inside forward())
        a = self.actor(batch["z_rl"], batch["s_p"], batch["ref_action"])
        q = self.critic.q1_only(batch["z_rl"], batch["s_p"], a)

        # BC regularization — target is always original ã, not dropped
        bc_loss = F.mse_loss(a, batch["ref_action"].flatten(1))
        loss = -q.mean() + self.beta * bc_loss

        self.actor_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        return {"actor_loss": loss.item(), "bc_loss": bc_loss.item(), "q_mean": q.mean().item()}

    def _soft_update_targets(self):
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def update(self, batch: dict) -> dict:
        """One full iteration: critic always, actor delayed, target soft update."""
        metrics = self.update_critic(batch)

        self.step_count += 1
        if self.step_count % self.actor_delay == 0:
            actor_metrics = self.update_actor(batch)
            metrics.update(actor_metrics)
            self._soft_update_targets()

        return metrics

    def sync_actor_target(self):
        """Full sync (used after BC pretrain)."""
        self.actor_target.load_state_dict(self.actor.state_dict())
