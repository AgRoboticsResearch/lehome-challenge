"""
Step-by-step RL episode runner for Stage 2 training.

Mirrors warmup_runner.py pattern exactly (from evaluation.py):
  - Step-by-step env interaction (one action per step)
  - env._get_rewards()  for reward (NOT env.step() return)  [evaluation.py:378]
  - env._get_success()   for success (doesn't trigger terminated)  [evaluation.py:372]
  - env._get_observations() every step  [evaluation.py:391]
  - Success → extra 50 settle steps → done
  - Timeout → done

Actor generates action chunks at chunk boundaries.
Between boundaries, actions are stepped through one at a time (open-loop within chunk).
VLA state computed at chunk boundaries for replay buffer transitions.
RL trainer updates at each chunk boundary.
"""

import random
import time
from typing import Optional

import numpy as np
import torch

from scripts.utils.common import stabilize_garment_after_reset


# ═══════════════════════════════════════════════════════════════════
# Helpers (same as warmup_runner)
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


def _actor_generate_chunk(actor, z_rl, s_p, ref_action, vla_hook, chunk_size, action_dim, device):
    """Actor forward -> (raw_actions_numpy, stored_actions_tensor)."""
    actor.train()
    with torch.no_grad():
        action_norm = actor(z_rl, s_p, ref_action)
    action_chunk_raw = vla_hook.denormalize_action(
        action_norm.view(1, chunk_size, action_dim)
    ).squeeze(0).cpu().numpy()  # (chunk_size, action_dim)
    action_chunk_stored = action_norm.view(1, chunk_size, action_dim).clone()
    return action_chunk_raw, action_chunk_stored


# ═══════════════════════════════════════════════════════════════════
# Main RL episode runner
# ═══════════════════════════════════════════════════════════════════


def run_rl_episodes(
    env,
    actor,
    vla_hook,
    stage1,
    replay_buffer,
    rl_trainer,
    cfg: dict,
    args,
    save_fn=None,
    garment_list: Optional[list[str]] = None,
):
    """Run step-by-step RL training episodes.

    Mirrors warmup_runner.py exactly, replacing MoE policy with RL actor.

    Pattern:
      1. At chunk boundary: VLA forward -> actor -> cache chunk of actions
      2. Per step: pop action from cache -> env.step() -> check reward/success
      3. At chunk boundary or done: store transition in replay buffer
      4. At chunk boundary: RL trainer update

    Env interaction (from evaluation.py:280-391):
      1. actor generates action chunk           -> numpy actions
      2. per step: env.step(action_tensor)       -> single action tensor
      3. env._get_success()                      -> explicit check, no terminated
      4. env._get_rewards()                      -> reward from env method
      5. obs = env._get_observations()           -> update obs EVERY step
    """
    device = torch.device(cfg["device"])
    env_device = cfg.get("env_device", "cpu")
    chunk_size = cfg["chunk_size"]
    gamma = cfg["gamma"]
    max_steps = cfg.get("max_episode_steps", 300)
    z_rl_dim = cfg["z_rl_dim"]
    state_dim = cfg["state_dim"]
    action_dim = cfg["action_dim"]
    num_episodes = cfg.get("total_episodes", 100)

    all_stats = []

    for ep in range(num_episodes):
        # ── Episode setup (evaluation.py:199-205) ──
        if garment_list:
            garment = random.choice(garment_list)
            env.switch_garment(garment, "Release")

        env.reset()
        stabilize_garment_after_reset(env, args)
        obs = env._get_observations()

        # VLA state for first chunk
        z_rl, ref_action, s_p = _compute_vla_state(
            obs, vla_hook, stage1, device, chunk_size
        )

        # Actor generates first action chunk
        action_chunk_raw, action_chunk_stored = _actor_generate_chunk(
            actor, z_rl, s_p, ref_action, vla_hook, chunk_size, action_dim, device
        )

        step_in_chunk = 0
        chunk_rewards = []
        episode_reward = 0.0
        episode_steps = 0
        success_flag = False
        extra_steps = 0
        episode_metrics = {"critic_loss": [], "actor_loss": [], "bc_loss": []}

        for step in range(max_steps):
            # ── Get action from cached chunk ──
            action_np = action_chunk_raw[step_in_chunk].copy()

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
            step_in_chunk += 1
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
            is_chunk_boundary = step_in_chunk == chunk_size
            if is_chunk_boundary or done:
                # Prepare stored action: zero-pad unexecuted tail
                stored = action_chunk_stored.clone()
                n_exec = len(chunk_rewards)
                if n_exec < chunk_size:
                    stored[:, n_exec:, :] = 0

                # Compute next VLA state (or zeros if done)
                if done:
                    next_z_rl = torch.zeros(1, z_rl_dim, device=device)
                    next_s_p = torch.zeros(1, state_dim, device=device)
                else:
                    next_z_rl, next_ref_action, next_s_p = _compute_vla_state(
                        obs, vla_hook, stage1, device, chunk_size
                    )

                # Discounted chunk return
                chunk_return = sum(
                    gamma ** t * r for t, r in enumerate(chunk_rewards)
                )

                # Store transition
                replay_buffer.add(
                    z_rl=z_rl,
                    s_p=s_p,
                    ref_action=ref_action,
                    action=stored,
                    reward=chunk_return,
                    next_z_rl=next_z_rl,
                    next_s_p=next_s_p,
                    done=done,
                )

                # ── RL updates ──
                batch_size = cfg["batch_size"]
                for _ in range(cfg.get("update_to_data_ratio", 5)):
                    if len(replay_buffer) >= batch_size:
                        batch = replay_buffer.sample(batch_size)
                        metrics = rl_trainer.update(batch)
                        for k in episode_metrics:
                            if k in metrics:
                                episode_metrics[k].append(metrics[k])

                # Prepare for next chunk
                if not done:
                    z_rl = next_z_rl
                    s_p = next_s_p
                    ref_action = next_ref_action
                    # Actor generates new action chunk
                    action_chunk_raw, action_chunk_stored = _actor_generate_chunk(
                        actor, z_rl, s_p, ref_action, vla_hook, chunk_size, action_dim, device
                    )

                step_in_chunk = 0
                chunk_rewards = []

            if done:
                break

        # ── Episode summary (evaluation.py:656) ──
        if success_flag:
            pass  # could track total successes if needed

        metrics_str = ""
        if episode_metrics:
            avg_m = {k: sum(v) / len(v) for k, v in episode_metrics.items() if v}
            metrics_str = ", " + ", ".join(f"{k}={v:.4f}" for k, v in avg_m.items()) if avg_m else ""

        print(
            f"  Ep {ep+1}/{num_episodes}: reward={episode_reward:.3f}, "
            f"steps={episode_steps}, "
            f"buffer={len(replay_buffer)}, "
            f"Success={success_flag}{metrics_str}"
        )

        stat = {
            "reward": episode_reward,
            "steps": episode_steps,
            "success": success_flag,
        }
        all_stats.append(stat)

        # Checkpoint
        if save_fn and (ep + 1) % cfg.get("save_freq", 50) == 0:
            save_fn(ep + 1, episode_reward, success_flag)

    return all_stats
