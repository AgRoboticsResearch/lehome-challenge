"""
Warmup runner for RL Token Stage 2.

Mirrors scripts/utils/evaluation.py:run_evaluation_loop() exactly.
Step-by-step env interaction (one action per step) — identical to eval.
MoE policy handles its own action queue (n_action_steps=12).
VLAStage2Hook provides z_rl/ref_action for replay buffer transitions.
"""

import random
import time
from typing import Optional

import numpy as np
import torch

from scripts.utils.common import stabilize_garment_after_reset


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _prepare_obs_batch(obs: dict, device: torch.device) -> dict:
    """Convert env obs (numpy) to VLAStage2Hook format (tensors).

    RGBA uint8 -> RGB float [0,1] -> CHW -> unsqueeze batch dim.
    State vectors pass through as-is.
    """
    batch = {}
    for key, value in obs.items():
        if not key.startswith("observation."):
            continue
        if isinstance(value, np.ndarray):
            t = torch.from_numpy(value.copy()).float()
            if t.ndim == 3 and t.shape[-1] >= 3:  # Image (H, W, C)
                t = t[..., :3]  # RGBA -> RGB
                t = t / 255.0  # [0, 255] -> [0, 1]
                t = t.permute(2, 0, 1)  # HWC -> CHW
            t = t.unsqueeze(0).to(device)
            batch[key] = t
    return batch


@torch.no_grad()
def _compute_vla_state(obs, vla_hook, stage1, device, chunk_size):
    """obs -> (z_rl, ref_action, s_p) for replay buffer."""
    batch = _prepare_obs_batch(obs, device)
    z_vlm, a_tilde = vla_hook.forward(batch)

    z_target = stage1.apply_keep_mask(z_vlm)
    z_rl = stage1.encoder(z_target)

    joint_pos = torch.as_tensor(
        obs["observation.state"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    s_p = vla_hook.normalize_state(joint_pos)

    ref_action = a_tilde[:, :chunk_size, :]
    return z_rl, ref_action, s_p


# ═══════════════════════════════════════════════════════════════════
# Main warmup runner
# ═══════════════════════════════════════════════════════════════════


def run_warmup_episodes(
    env,
    moe_policy,
    vla_hook,
    stage1,
    replay_buffer,
    cfg: dict,
    args,
    garment_list: Optional[list[str]] = None,
):
    """Run warmup episodes mirroring evaluation.py step-by-step.

    MoE policy generates actions (identical to eval pipeline).
    VLAStage2Hook computes z_rl/ref_action for replay buffer.

    Env interaction pattern (from evaluation.py:280-391):
      1. policy.select_action(obs)       -> numpy action
      2. torch -> env.step(action)       -> single action tensor
      3. env._get_success()              -> explicit check, no terminated
      4. env._get_rewards()              -> reward from env method
      5. obs = env._get_observations()   -> update obs EVERY step
    """
    device = torch.device(cfg["device"])
    env_device = cfg.get("env_device", "cpu")
    chunk_size = cfg["chunk_size"]
    gamma = cfg["gamma"]
    max_steps = cfg.get("max_episode_steps", 300)
    z_rl_dim = cfg["z_rl_dim"]
    state_dim = cfg["state_dim"]
    action_dim = cfg["action_dim"]
    num_episodes = cfg["warmup_episodes"]

    start_time = time.time()
    total_successes = 0

    for ep in range(num_episodes):
        # ── Episode setup (evaluation.py:199-205) ──
        if garment_list:
            garment = random.choice(garment_list)
            env.switch_garment(garment, "Release")

        env.reset()
        moe_policy.reset()
        stabilize_garment_after_reset(env, args)
        obs = env._get_observations()

        # VLA state for first chunk
        z_rl, ref_action, s_p = _compute_vla_state(
            obs, vla_hook, stage1, device, chunk_size
        )

        chunk_actions_raw = []  # denormalized MoE actions
        chunk_rewards = []
        episode_reward = 0.0
        episode_steps = 0
        success_flag = False
        extra_steps = 0

        for step in range(max_steps):
            # ── Get action (evaluation.py:312) ──
            action_np = moe_policy.select_action(obs)

            # ── Step environment (evaluation.py:331, 351) ──
            action_tensor = (
                torch.from_numpy(action_np).float().to(env_device).unsqueeze(0)
            )
            env.step(action_tensor)

            # ── Check success (evaluation.py:372-375) ──
            if not success_flag:
                success_val = env._get_success()
                if (
                    success_val.item()
                    if torch.is_tensor(success_val)
                    else bool(success_val)
                ):
                    success_flag = True
                    extra_steps = 50  # settle after success

            # ── Get reward (evaluation.py:378-383) ──
            reward_value = env._get_rewards()
            reward = (
                reward_value.item()
                if torch.is_tensor(reward_value)
                else float(reward_value)
            )
            episode_reward += reward

            # ── Update observation (evaluation.py:391) ──
            obs = env._get_observations()
            episode_steps += 1

            # ── Collect for replay buffer ──
            chunk_actions_raw.append(action_np.copy())
            chunk_rewards.append(reward)

            # ── Determine if episode is done ──
            done = False
            if success_flag:
                extra_steps -= 1
                if extra_steps <= 0:
                    done = True
            if step >= max_steps - 1 and not done:
                done = True

            # ── Store transition at chunk boundary or episode end ──
            is_chunk_boundary = len(chunk_actions_raw) == chunk_size
            if is_chunk_boundary or done:
                # Save current chunk's ref_action before recomputing
                chunk_ref_action = ref_action

                # Compute next VLA state (or zeros if done)
                if done:
                    next_z_rl = torch.zeros(1, z_rl_dim, device=device)
                    next_s_p = torch.zeros(1, state_dim, device=device)
                    next_ref_action = None
                else:
                    next_z_rl, next_ref_action, next_s_p = _compute_vla_state(
                        obs, vla_hook, stage1, device, chunk_size
                    )

                # Normalize executed actions for replay buffer
                n_exec = len(chunk_actions_raw)
                actions_np = np.stack(chunk_actions_raw)  # (n_exec, action_dim)
                actions_tensor = torch.from_numpy(actions_np).float().to(device)
                stored_action = vla_hook.normalize_action(actions_tensor).unsqueeze(
                    0
                )  # (1, n_exec, action_dim)

                # Pad partial chunks to chunk_size
                if n_exec < chunk_size:
                    padded = torch.zeros(
                        1, chunk_size, action_dim, device=device
                    )
                    padded[:, :n_exec, :] = stored_action
                    stored_action = padded

                # Discounted chunk return
                chunk_return = sum(
                    gamma ** t * r for t, r in enumerate(chunk_rewards)
                )

                replay_buffer.add(
                    z_rl=z_rl,
                    s_p=s_p,
                    ref_action=chunk_ref_action,
                    action=stored_action,
                    reward=chunk_return,
                    next_z_rl=next_z_rl,
                    next_s_p=next_s_p,
                    done=done,
                )

                # Prepare for next chunk
                z_rl = next_z_rl
                s_p = next_s_p
                if next_ref_action is not None:
                    ref_action = next_ref_action
                chunk_actions_raw = []
                chunk_rewards = []

            if done:
                break

        # ── Episode summary (evaluation.py:656) ──
        if success_flag:
            total_successes += 1
        print(
            f"  Ep {ep+1}/{num_episodes}: reward={episode_reward:.3f}, "
            f"steps={episode_steps}, Success={success_flag}, "
            f"buffer={len(replay_buffer)}"
        )

    elapsed = time.time() - start_time
    print(
        f"\n  Warmup complete: {len(replay_buffer)} transitions, "
        f"{total_successes}/{num_episodes} successes, {elapsed:.1f}s"
    )
