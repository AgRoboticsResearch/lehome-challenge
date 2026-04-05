"""
Chunk-level episode runner for RL Token Stage 2 training.

Reuses correct env interaction patterns from evaluation.py:
  - env._get_rewards()  for reward (NOT env.step() return)  [evaluation.py:378]
  - env._get_success()   for success (not triggered by env.step())  [evaluation.py:372]
  - env.switch_garment() + stabilize + _get_observations()  [evaluation.py:862-866,201,205]
"""

import abc
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from scripts.utils.common import stabilize_garment_after_reset


# ═══════════════════════════════════════════════════════════════════
# Low-level chunk execution — the core fix
# ═══════════════════════════════════════════════════════════════════


def _execute_chunk(env, action_raw: np.ndarray, chunk_size: int):
    """Execute action chunk open-loop with correct reward/success handling.

    Mirrors evaluation.py patterns:
      - reward from env._get_rewards() (line 378), NOT env.step() return
      - explicit success check (line 372), success doesn't trigger terminated
      - single action tensor per step (line 351)
    """
    rewards = []
    done = False
    success = False
    for t in range(chunk_size):
        action_tensor = torch.from_numpy(action_raw[t]).float().unsqueeze(0)
        env.step(action_tensor)

        # Reward: _get_rewards() handles step_interval caching correctly
        reward_value = env._get_rewards()
        r = reward_value.item() if torch.is_tensor(reward_value) else float(reward_value)
        rewards.append(r)

        # Terminated: timeout check
        dones = env._get_dones()
        terminated = dones[0] if isinstance(dones, tuple) else dones
        if terminated.item() if torch.is_tensor(terminated) else bool(terminated):
            done = True
            break

        # Success: explicit check — does NOT set terminated (evaluation.py:372)
        success_val = env._get_success()
        if success_val.item() if torch.is_tensor(success_val) else bool(success_val):
            done = True
            success = True
            break

    return rewards, done, success


# ═══════════════════════════════════════════════════════════════════
# Chunk result & policy protocol
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ChunkResult:
    """Output from a chunk policy's get_chunk() call."""
    action_raw: np.ndarray       # (chunk_size, action_dim) — denormalized, for env.step()
    z_rl: torch.Tensor           # (1, z_rl_dim) — for replay buffer
    s_p: torch.Tensor            # (1, state_dim) — for replay buffer
    ref_action: torch.Tensor     # (1, chunk_size, action_dim) — VLA ã (normalized)
    stored_action: torch.Tensor  # (1, chunk_size, action_dim) — executed action (normalized)


class BaseChunkPolicy(abc.ABC):
    """Abstract base for chunk-level policies.

    Includes VLM output caching: get_state() caches (z_rl, a_tilde, s_p)
    so the next get_chunk() can reuse them, avoiding a redundant VLM forward.
    """

    def __init__(self):
        self._cached_vlm: Optional[tuple] = None  # (z_rl, a_tilde, s_p)

    def reset(self):
        self._cached_vlm = None

    @abc.abstractmethod
    def get_chunk(self, obs: dict) -> ChunkResult:
        pass

    @abc.abstractmethod
    def get_state(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Get (z_rl, s_p) and cache VLM output for next get_chunk()."""
        pass

    def _try_pop_cache(self):
        """Pop cached VLM output. Returns None if cache empty."""
        cache = self._cached_vlm
        self._cached_vlm = None
        return cache

    # ── Shared obs processing ──

    def _prepare_batch(self, obs: dict) -> dict:
        """Convert env obs to VLAStage2Hook format.

        Same as prepare_obs_batch() in old training script:
          RGBA uint8 → RGB float [0,1] → CHW → unsqueeze batch dim
        """
        batch = {}
        for key, value in obs.items():
            if not key.startswith("observation."):
                continue
            if isinstance(value, np.ndarray):
                t = torch.from_numpy(value.copy()).float()
                if t.ndim == 3 and t.shape[-1] >= 3:  # Image (H, W, C)
                    t = t[..., :3]          # RGBA → RGB
                    t = t / 255.0           # [0, 255] → [0, 1]
                    t = t.permute(2, 0, 1)  # HWC → CHW
                t = t.unsqueeze(0).to(self.device)
                batch[key] = t
        return batch

    @torch.no_grad()
    def _process_obs(self, obs: dict):
        """obs → (z_rl, a_tilde, s_p). Reuses VLAStage2Hook + RLTokenStage1."""
        batch = self._prepare_batch(obs)
        z_vlm, a_tilde = self.vla_hook.forward(batch)

        z_target = self.stage1.apply_keep_mask(z_vlm)
        z_rl = self.stage1.encoder(z_target)

        joint_pos = torch.as_tensor(
            obs["observation.state"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        s_p = self.vla_hook.normalize_state(joint_pos)

        return z_rl, a_tilde, s_p


# ═══════════════════════════════════════════════════════════════════
# WarmupPolicy — VLA only (Phase 4)
# ═══════════════════════════════════════════════════════════════════


class WarmupPolicy(BaseChunkPolicy):
    """VLA-only policy for warmup. Actions = VLA reference directly."""

    def __init__(self, vla_hook, stage1, device, chunk_size):
        super().__init__()
        self.vla_hook = vla_hook
        self.stage1 = stage1
        self.device = device
        self.chunk_size = chunk_size

    @torch.no_grad()
    def get_chunk(self, obs: dict) -> ChunkResult:
        cache = self._try_pop_cache()
        if cache is not None:
            z_rl, a_tilde, s_p = cache
        else:
            z_rl, a_tilde, s_p = self._process_obs(obs)
        action_chunk_norm = a_tilde[:, :self.chunk_size, :]
        action_raw = self.vla_hook.denormalize_action(
            action_chunk_norm
        ).squeeze(0).cpu().numpy()
        return ChunkResult(
            action_raw=action_raw,
            z_rl=z_rl, s_p=s_p,
            ref_action=a_tilde[:, :self.chunk_size, :],
            stored_action=action_chunk_norm.clone(),
        )

    @torch.no_grad()
    def get_state(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        z_rl, a_tilde, s_p = self._process_obs(obs)
        self._cached_vlm = (z_rl, a_tilde, s_p)  # Cache for next get_chunk
        return z_rl, s_p


# ═══════════════════════════════════════════════════════════════════
# DecoupledWarmupPolicy — debug: MoE for actions, VLA hook for z_rl
# ═══════════════════════════════════════════════════════════════════


class DecoupledWarmupPolicy(BaseChunkPolicy):
    """Debug warmup policy that decouples action generation from z_rl computation.

    Action path: Uses MoESmolVLAPolicy.select_action() (identical to eval pipeline).
    z_rl path:   Uses VLAStage2Hook + RLTokenStage1 (for replay buffer).

    Purpose: Isolate whether VLAStage2Hook's action generation causes zero success.
    If this policy succeeds but WarmupPolicy doesn't, the VLAStage2Hook action
    generation is broken. If both fail, the problem is elsewhere (env, success
    detection, etc.).
    """

    def __init__(self, moe_policy, vla_hook, stage1, device, chunk_size):
        super().__init__()
        self.moe_policy = moe_policy    # MoESmolVLAPolicy (eval pipeline)
        self.vla_hook = vla_hook
        self.stage1 = stage1
        self.device = device
        self.chunk_size = chunk_size

    def reset(self):
        super().reset()
        self.moe_policy.reset()

    @torch.no_grad()
    def get_chunk(self, obs: dict) -> ChunkResult:
        # ── z_rl path: VLAStage2Hook (unchanged from WarmupPolicy) ──
        cache = self._try_pop_cache()
        if cache is not None:
            z_rl, a_tilde, s_p = cache
        else:
            z_rl, a_tilde, s_p = self._process_obs(obs)

        # ── Action path: MoE policy (same as eval!) ──
        # MoE policy has internal action queue (n_action_steps=12).
        # Calling select_action chunk_size times drains the queue.
        # First call triggers VLM forward; subsequent calls pop from queue.
        actions = []
        for _ in range(self.chunk_size):
            action_np = self.moe_policy.select_action(obs)
            actions.append(action_np.copy())
        action_raw = np.stack(actions)  # (chunk_size, action_dim)

        # ref_action from VLA hook (normalized) — for replay buffer
        ref_action = a_tilde[:, :self.chunk_size, :]

        # stored_action: MoE actions converted to normalized space
        action_tensor = torch.from_numpy(action_raw).float().to(self.device)
        stored_action = self.vla_hook.normalize_action(action_tensor).unsqueeze(0)

        return ChunkResult(
            action_raw=action_raw,
            z_rl=z_rl, s_p=s_p,
            ref_action=ref_action,
            stored_action=stored_action,
        )

    @torch.no_grad()
    def get_state(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        z_rl, a_tilde, s_p = self._process_obs(obs)
        self._cached_vlm = (z_rl, a_tilde, s_p)
        return z_rl, s_p


# ═══════════════════════════════════════════════════════════════════
# RLActorPolicy — actor + VLA reference (Phase 5)
# ═══════════════════════════════════════════════════════════════════


class RLActorPolicy(BaseChunkPolicy):
    """RL actor policy with VLA reference conditioning."""

    def __init__(self, actor, vla_hook, stage1, device, chunk_size, action_dim):
        super().__init__()
        self.actor = actor
        self.vla_hook = vla_hook
        self.stage1 = stage1
        self.device = device
        self.chunk_size = chunk_size
        self.action_dim = action_dim

    @torch.no_grad()
    def get_chunk(self, obs: dict) -> ChunkResult:
        # Check cache first (from previous get_state call on same obs)
        cache = self._try_pop_cache()
        if cache is not None:
            z_rl, a_tilde, s_p = cache
        else:
            z_rl, a_tilde, s_p = self._process_obs(obs)

        # Actor forward under no_grad — experience collection, no backprop needed
        # Dropout still active because actor.train() is set, just no gradient graph built
        self.actor.train()
        action_norm = self.actor(z_rl, s_p, a_tilde[:, :self.chunk_size, :])
        action_raw = self.vla_hook.denormalize_action(
            action_norm.view(1, self.chunk_size, self.action_dim)
        ).squeeze(0).cpu().numpy()

        return ChunkResult(
            action_raw=action_raw,
            z_rl=z_rl, s_p=s_p,
            ref_action=a_tilde[:, :self.chunk_size, :],
            stored_action=action_norm.view(1, self.chunk_size, self.action_dim).clone(),
        )

    @torch.no_grad()
    def get_state(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        z_rl, a_tilde, s_p = self._process_obs(obs)
        self._cached_vlm = (z_rl, a_tilde, s_p)  # Cache for next get_chunk
        return z_rl, s_p


# ═══════════════════════════════════════════════════════════════════
# Main episode runner
# ═══════════════════════════════════════════════════════════════════


def run_chunk_episodes(
    env,
    policy: BaseChunkPolicy,
    num_episodes: int,
    cfg: dict,
    args,
    replay_buffer=None,
    garment_list: Optional[list[str]] = None,
    rl_trainer=None,
    save_fn=None,
):
    """Run chunk-level training episodes with correct env interaction.

    Args:
        env: Isaac Sim garment environment
        policy: ChunkPolicy (WarmupPolicy or RLActorPolicy)
        num_episodes: Number of episodes to run
        cfg: Config dict (chunk_size, gamma, max_episode_steps, etc.)
        args: Namespace for stabilize_garment_after_reset
        replay_buffer: If provided, stores transitions
        garment_list: If provided, randomly samples garment each episode
        rl_trainer: If provided, does RL updates after each chunk (Phase 5)
        save_fn: Optional callback(ep_num, reward, success) for checkpoints

    Returns:
        List of episode stats dicts
    """
    chunk_size = cfg["chunk_size"]
    gamma = cfg["gamma"]
    max_steps = cfg.get("max_episode_steps", 300)
    device = cfg["device"]
    z_rl_dim = cfg["z_rl_dim"]
    state_dim = cfg["state_dim"]

    all_stats = []

    for ep in range(num_episodes):
        # ── Episode setup (from eval pattern: evaluation.py:862-866, 201, 205) ──
        if garment_list:
            garment = random.choice(garment_list)
            env.switch_garment(garment, "Release")
        env.reset()
        stabilize_garment_after_reset(env, args)
        obs = env._get_observations()
        policy.reset()

        episode_reward = 0
        episode_steps = 0
        episode_metrics = {"critic_loss": [], "actor_loss": [], "bc_loss": []} if rl_trainer else {}

        while episode_steps < max_steps:
            # ── Policy decides action chunk ──
            chunk = policy.get_chunk(obs)

            # ── Execute chunk with correct reward/success ──
            rewards, done, success = _execute_chunk(env, chunk.action_raw, chunk_size)
            n_exec = len(rewards)
            chunk_return = sum(gamma ** t * r for t, r in enumerate(rewards))
            episode_reward += sum(rewards)
            episode_steps += n_exec

            # Handle partial chunk execution
            stored = chunk.stored_action.clone()
            if n_exec < chunk_size:
                stored[:, n_exec:] = 0

            # ── Get next state ──
            if done:
                next_z_rl = torch.zeros(1, z_rl_dim, device=device)
                next_s_p = torch.zeros(1, state_dim, device=device)
            else:
                obs = env._get_observations()  # Update obs once (reused by next get_chunk via cache)
                next_z_rl, next_s_p = policy.get_state(obs)

            # ── Store transition ──
            if replay_buffer is not None:
                replay_buffer.add(
                    z_rl=chunk.z_rl, s_p=chunk.s_p,
                    ref_action=chunk.ref_action,
                    action=stored,
                    reward=chunk_return,
                    next_z_rl=next_z_rl, next_s_p=next_s_p,
                    done=done,
                )

            # ── RL updates (Phase 5 only) ──
            if rl_trainer is not None:
                batch_size = cfg["batch_size"]
                for _ in range(cfg.get("update_to_data_ratio", 1)):
                    if len(replay_buffer) >= batch_size:
                        batch = replay_buffer.sample(batch_size)
                        metrics = rl_trainer.update(batch)
                        for k in episode_metrics:
                            if k in metrics:
                                episode_metrics[k].append(metrics[k])

            if done:
                break
            # obs already updated at line ~312 (env._get_observations + get_state cache)

        # ── Timeout: mark last transition as done if loop exited without it ──
        if not done and replay_buffer is not None and replay_buffer.size > 0:
            # Patch the last stored transition: done=True, next state = zeros
            last_idx = (replay_buffer.ptr - 1) % replay_buffer.capacity
            replay_buffer.done[last_idx] = True
            replay_buffer.next_z_rl[last_idx] = 0
            replay_buffer.next_s_p[last_idx] = 0

        # ── Episode summary (from eval pattern: evaluation.py:656) ──
        final_success = env._get_success()
        is_success = final_success.item() if torch.is_tensor(final_success) else bool(final_success)
        avg_r = episode_reward / max(episode_steps, 1)

        stat = {
            "reward": episode_reward, "steps": episode_steps,
            "avg_r_step": avg_r, "success": is_success,
        }
        all_stats.append(stat)

        # Format like eval output
        metrics_str = ""
        if episode_metrics:
            avg_m = {k: sum(v) / len(v) for k, v in episode_metrics.items() if v}
            metrics_str = ", " + ", ".join(f"{k}={v:.4f}" for k, v in avg_m.items()) if avg_m else ""

        print(f"  Ep {ep+1}/{num_episodes}: reward={episode_reward:.3f}, "
              f"steps={episode_steps}, avg_r/step={avg_r:.4f}, "
              f"buffer={len(replay_buffer) if replay_buffer else 'N/A'}, "
              f"Success={is_success}{metrics_str}")

        # Checkpoint
        if save_fn and (ep + 1) % cfg.get("save_freq", 50) == 0:
            save_fn(ep + 1, episode_reward, is_success)

    return all_stats
