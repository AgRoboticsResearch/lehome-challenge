"""
Stage 2 RL Token Training: Online TD3+BC with VLAStage2Hook.

Usage:
    python -m scripts.train_rl_token_stage2 --config configs/train_rl_stage2.yaml

Requires: Isaac Sim running, SmolVLA checkpoint, Stage 1 RL Token checkpoint.

Pipeline:
    Phase 4:   Warmup — fill ReplayBuffer with VLA actions (no RL)
    Phase 4.5: BC pretrain — initialize actor to mimic VLA
    Phase 5:   Online RL — TD3+BC training loop
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from lehome.models.rl_stage2 import (
    RLActor,
    TwinCritic,
    RLTTrainer,
    ReplayBuffer,
    SimpleNormalizer,
)
from lehome.models.rl_token import RLTokenStage1
from lehome.models.vla_stage2_hook import VLAStage2Hook


import yaml


import os


os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


os.environ.setdefault("TOKENIZERS_PARALLEL", "1")


# ═══════════════════════════════════════════════════════════════════
# Observation preprocessing
# ═══════════════════════════════════════════════════════════════════


def prepare_obs_batch(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Convert env._get_observations() output to VLAStage2Hook format.

    env output:
        observation.images.*_rgb: (H, W, 4) uint8 RGBA numpy  ← Isaac Sim "rgb" = 4ch
        observation.state:         (12,)     float32 numpy
    hook expects:
        observation.images.*_rgb: (1, C, H, W) float32 tensor [0, 1]
        observation.state:         (1, 12)     float32 tensor
    """
    batch = {}
    for key, value in obs.items():
        if not key.startswith("observation."):
            continue
        if isinstance(value, np.ndarray):
            t = torch.from_numpy(value.copy()).float()
            if t.ndim == 3 and t.shape[-1] >= 3:  # Image (H, W, C)
                t = t[..., :3]                      # RGBA → RGB
                t = t / 255.0                       # [0, 255] → [0, 1]
                t = t.permute(2, 0, 1)              # (H, W, C) → (C, H, W)
            t = t.unsqueeze(0).to(device)           # +batch dim, to GPU
            batch[key] = t
    return batch


@torch.no_grad()
def process_observation(obs_dict, vla_hook, stage1, normalizer, device):
    """Full pipeline: env obs → (z_rl, a_tilde, s_p).

    Returns:
        z_rl:    (1, 960) — RL Token state
        a_tilde: (1, 50, 12) — VLA reference action chunk (normalized)
        s_p:     (1, 12) — normalized joint positions
    """
    batch = prepare_obs_batch(obs_dict, device)
    z_vlm, a_tilde = vla_hook.forward(batch)

    z_target = stage1.apply_keep_mask(z_vlm)
    z_rl = stage1.encoder(z_target)

    joint_pos = torch.as_tensor(
        obs_dict["observation.state"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    s_p = normalizer.normalize_state(joint_pos)

    return z_rl, a_tilde, s_p


def execute_chunk(env, action_raw: np.ndarray, max_steps: int, gamma: float):
    """Execute action chunk open-loop on return rewards and done flag."""
    rewards = []
    done = False
    for t in range(max_steps):
        action_tensor = torch.from_numpy(action_raw[t]).float().unsqueeze(0)
        obs, reward, terminated, truncated, info = env.step(action_tensor)
        done = terminated.item() if torch.is_tensor(terminated) else bool(terminated)
        r = reward.item() if torch.is_tensor(reward) else float(reward)
        rewards.append(r)
        if done:
            break
    return rewards, done, obs


def compute_chunk_return(rewards: list[float], gamma: float) -> float:
    return sum(gamma ** t * r for t, r in enumerate(rewards))


# ═══════════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════════


def train(cfg: dict, simulation_app):
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from scripts.utils.common import stabilize_garment_after_reset

    device = torch.device(cfg["device"])
    env_device = cfg.get("env_device", "cpu")
    chunk_size = cfg["chunk_size"]
    gamma = cfg["gamma"]

    # ── Phase 1: Load frozen Components ──
    print("=" * 60)
    print("Phase 1: Loading Frozen Components")
    print("=" * 60)

    normalizer = SimpleNormalizer(cfg["dataset_stats_path"], device)

    vla_hook = VLAStage2Hook(
        pretrained_path=cfg["smolvla_pretrained_path"],
        device=str(device),
        task_description=cfg.get("task_description", "fold the garment"),
    )

    stage1 = RLTokenStage1()
    ckpt = torch.load(cfg["rl_token_stage1_path"], map_location=device, weights_only=False)
    stage1.load_state_dict(ckpt["model_state_dict"])
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad = False
    stage1.to(device)

    print(f"  Normalizer: {cfg['dataset_stats_path']}")
    print(f"  VLA hook: {cfg['smolvla_pretrained_path']}")
    print(f"  Stage 1: {cfg['rl_token_stage1_path']}")

    # ── Phase 2: Create Trainable Components ──
    print("\n" + "=" * 60)
    print("Phase 2: Creating Trainable Components")
    print("=" * 60)

    actor = RLActor(
        z_rl_dim=cfg["z_rl_dim"],
        state_dim=cfg["state_dim"],
        chunk_size=chunk_size,
        action_dim=cfg["action_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        fixed_std=cfg.get("fixed_std", 0.0067),
        ref_dropout=cfg.get("ref_dropout", 0.5),
    ).to(device)

    critic = TwinCritic(
        z_rl_dim=cfg["z_rl_dim"],
        state_dim=cfg["state_dim"],
        chunk_size=chunk_size,
        action_dim=cfg["action_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to(device)

    replay = ReplayBuffer(
        capacity=cfg["replay_capacity"],
        z_rl_dim=cfg["z_rl_dim"],
        state_dim=cfg["state_dim"],
        chunk_size=chunk_size,
        action_dim=cfg["action_dim"],
        device=device,
    )

    trainer = RLTTrainer(
        actor=actor,
        critic=critic,
        device=device,
        actor_lr=cfg["actor_lr"],
        critic_lr=cfg["critic_lr"],
        gamma=gamma,
        tau=cfg["tau"],
        beta=cfg["beta"],
        target_noise_std=cfg.get("target_noise_std", 0.2),
        noise_clip=cfg.get("noise_clip", 0.5),
        chunk_size=chunk_size,
        actor_delay=cfg["actor_delay"],
        grad_clip=cfg["grad_clip"],
    )

    total_params = sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in critic.parameters())
    print(f"  Actor: {sum(p.numel() for p in actor.parameters()):,} params")
    print(f"  Critic: {sum(p.numel() for p in critic.parameters()):,} params")
    print(f"  ReplayBuffer: {cfg['replay_capacity']:,} capacity")

    # ── Phase 3: Create Environment ──
    print("\n" + "=" * 60)
    print("Phase 3: Creating Isaac Sim Environment")
    print("=" * 60)

    args_namespace = argparse.Namespace(**{
        "task": cfg.get("task", "LeHome-BiSO101-Direct-Garment-v2"),
        "device": env_device,
        "seed": cfg.get("seed", 42),
    "use_random_seed": False,
        "garment_cfg_base_path": cfg.get("garment_cfg_base_path", "Assets/objects/Challenge_Garment"),
        "particle_cfg_path": cfg.get("particle_cfg_path", "source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml"),
        "teleop_device": "keyboard",
    })
    env_cfg = parse_env_cfg(args_namespace.task, device=env_device)
    env_cfg.sim.use_fabric = False
    env_cfg.use_random_seed = False
    env_cfg.seed = cfg.get("seed", 42)
    env_cfg.garment_cfg_base_path = args_namespace.garment_cfg_base_path
    env_cfg.particle_cfg_path = args_namespace.particle_cfg_path
    env_cfg.garment_name = cfg["garment_name"]

    env = gym.make(args_namespace.task, cfg=env_cfg).unwrapped
    env.initialize_obs()
    print(f"  Env: {args_namespace.task}, garment={cfg['garment_name']}")

    # ── Phase 4: Warmup ──
    print("\n" + "=" * 60)
    print(f"Phase 4: Warmup ({cfg['warmup_episodes']} episodes)")
    print("=" * 60)

    warmup_start = time.time()
    for ep in range(cfg["warmup_episodes"]):
        obs, info = env.reset()
        stabilize_garment_after_reset(env, args_namespace)

        z_rl, a_tilde, s_p = process_observation(obs, vla_hook, stage1, normalizer, device)
        episode_reward = 0

        while True:
            # Use VLA actions directly
            action_chunk_norm = a_tilde[:, :chunk_size, :]
            action_raw_np = normalizer.denormalize_action(action_chunk_norm).squeeze(0).cpu().numpy()

            rewards, done, last_obs = execute_chunk(env, action_raw_np, chunk_size, gamma)
            n_exec = len(rewards)
            chunk_return = compute_chunk_return(rewards, gamma)
            episode_reward += sum(rewards)

            action_stored = action_chunk_norm.clone()
            if n_exec < chunk_size:
                action_stored[:, n_exec:] = 0

            if done:
                next_z_rl = torch.zeros(1, cfg["z_rl_dim"], device=device)
                next_s_p = torch.zeros(1, cfg["state_dim"], device=device)
            else:
                last_obs_dict = env._get_observations()
                next_z_rl, _, next_s_p = process_observation(
                    last_obs_dict, vla_hook, stage1, normalizer, device
                )

            replay.add(
                z_rl=z_rl, s_p=s_p,
                ref_action=a_tilde[:, :chunk_size, :],
                action=action_stored,
                reward=chunk_return,
                next_z_rl=next_z_rl, next_s_p=next_s_p,
                done=done,
            )

            if done:
                break

            obs = env._get_observations()
            z_rl, a_tilde, s_p = process_observation(obs, vla_hook, stage1, normalizer, device)

        print(f"  Ep {ep+1}/{cfg['warmup_episodes']}: reward={episode_reward:.3f}, buffer={len(replay)}")

    warmup_time = time.time() - warmup_start
    print(f"  Warmup done: {len(replay)} transitions in {warmup_time:.1f}s")

    # ── Phase 4.5: BC Pretrain Actor ──
    print("\n" + "=" * 60)
    print("Phase 4.5: BC Pretrain (actor warm start from VLA)")
    print("=" * 60)

    bc_optim = Adam(actor.parameters(), lr=cfg["bc_lr"])
    actor.train()

    for epoch in range(cfg["bc_pretrain_epochs"]):
        total_loss = 0
        n_batches = 0
        for _ in range(cfg["bc_batches_per_epoch"]):
            batch = replay.sample(cfg["batch_size"])
            # Full ref, no dropout during BC pretrain
            a_pred = actor(batch["z_rl"], batch["s_p"], batch["ref_action"])
            loss = F.mse_loss(a_pred, batch["ref_action"].flatten(1))

            bc_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg["grad_clip"])
            bc_optim.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  BC epoch {epoch+1}/{cfg['bc_pretrain_epochs']}: loss = {avg_loss:.6f}")
        if avg_loss < cfg["bc_loss_threshold"]:
            print(f"  BC converged at epoch {epoch+1}")
            break

    trainer.sync_actor_target()
    print("  Actor target synced with BC-pretrained weights")

    # ── Phase 5: Online RL ──
    print("\n" + "=" * 60)
    print(f"Phase 5: Online RL ({cfg['total_episodes']} episodes)")
    print("=" * 60)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_reward = -float("inf")

    for ep in range(cfg["total_episodes"]):
        obs, info = env.reset()
        stabilize_garment_after_reset(env, args_namespace)

        z_rl, a_tilde, s_p = process_observation(obs, vla_hook, stage1, normalizer, device)
        episode_reward = 0
        episode_metrics = {"critic_loss": [], "actor_loss": [], "bc_loss": []}

        while True:
            # Actor forward with ref dropout
            actor.train()
            action_norm = actor(z_rl, s_p, a_tilde[:, :chunk_size, :])
            action_raw = normalizer.denormalize_action(
                action_norm.view(1, chunk_size, cfg["action_dim"])
            ).squeeze(0).cpu().numpy()

            rewards, done, last_obs = execute_chunk(env, action_raw, chunk_size, gamma)
            n_exec = len(rewards)
            chunk_return = compute_chunk_return(rewards, gamma)
            episode_reward += sum(rewards)

            action_stored = action_norm.detach().view(1, chunk_size, cfg["action_dim"]).clone()
            if n_exec < chunk_size:
                action_stored[:, n_exec:] = 0

            if done:
                next_z_rl = torch.zeros(1, cfg["z_rl_dim"], device=device)
                next_s_p = torch.zeros(1, cfg["state_dim"], device=device)
            else:
                last_obs_dict = env._get_observations()
                next_z_rl, _, next_s_p = process_observation(
                    last_obs_dict, vla_hook, stage1, normalizer, device
                )

            replay.add(
                z_rl=z_rl, s_p=s_p,
                ref_action=a_tilde[:, :chunk_size, :],
                action=action_stored,
                reward=chunk_return,
                next_z_rl=next_z_rl, next_s_p=next_s_p,
                done=done,
            )

            # RL updates
            for _ in range(cfg["update_to_data_ratio"]):
                if len(replay) >= cfg["batch_size"]:
                    batch = replay.sample(cfg["batch_size"])
                    metrics = trainer.update(batch)
                    for k in episode_metrics:
                        if k in metrics:
                            episode_metrics[k].append(metrics[k])

            if done:
                break

            obs = env._get_observations()
            z_rl, a_tilde, s_p = process_observation(obs, vla_hook, stage1, normalizer, device)

        avg_metrics = {k: sum(v) / len(v) for k, v in episode_metrics.items() if v}
        print(f"  Ep {ep+1}: reward={episode_reward:.3f}, buffer={len(replay)}, "
              + ", ".join(f"{k}={v:.4f}" for k, v in avg_metrics.items()))

        if (ep + 1) % cfg["save_freq"] == 0:
            ckpt_path = output_dir / f"episode_{ep+1}.pt"
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "episode": ep + 1,
                "episode_reward": episode_reward,
            }, ckpt_path)
            print(f"  Checkpoint: {ckpt_path}")

            if episode_reward > best_reward:
                best_reward = episode_reward
                best_path = output_dir / "best_actor.pt"
                torch.save(actor.state_dict(), best_path)
                print(f"  New best: {best_reward:.3f}")

    print("\n" + "=" * 60)
    print(f"Training complete. Best reward: {best_reward:.3f}")
    print(f"Checkpoints: {output_dir}")
    print("=" * 60)


def main():
    import multiprocessing
    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Stage 2 RL Token Training")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

    # Launch Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import lehome.tasks.bedroom  # noqa: F401 — register tasks

        train(cfg, simulation_app)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

