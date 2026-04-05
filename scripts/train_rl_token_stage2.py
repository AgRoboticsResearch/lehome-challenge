"""
Stage 2 RL Token Training: Online TD3+BC with VLAStage2Hook.

Usage:
    python -m scripts.train_rl_token_stage2 --config configs/train_rl_stage2.yaml

Requires: Isaac Sim running, SmolVLA checkpoint, Stage 1 RL Token checkpoint.

Pipeline:
    Phase 4:   Warmup — fill ReplayBuffer with VLA actions (no RL)
    Phase 4.5: BC pretrain — initialize actor to mimic VLA
    Phase 5:   Online RL — TD3+BC training loop

Env interaction is handled by chunk_runner which reuses correct patterns
from evaluation.py (env._get_rewards, env._get_success, stabilize, etc).
"""

import argparse
import random
import time
from pathlib import Path

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
from scripts.utils.chunk_runner import (
    WarmupPolicy,
    DecoupledWarmupPolicy,
    RLActorPolicy,
    run_chunk_episodes,
)

import yaml
import os

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLEL", "1")


# ═══════════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════════


def train(cfg: dict, simulation_app):
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    device = torch.device(cfg["device"])
    env_device = cfg.get("env_device", "cpu")
    chunk_size = cfg["chunk_size"]

    # ── Phase 1: Load frozen Components ──
    print("=" * 60)
    print("Phase 1: Loading Frozen Components")
    print("=" * 60)

    normalizer = SimpleNormalizer(cfg["dataset_stats_path"], device)

    vla_hook = VLAStage2Hook(
        pretrained_path=cfg["smolvla_pretrained_path"],
        device=str(device),
        task_description=cfg.get("task_description", "fold the garment"),
        dataset_stats_path=cfg["dataset_stats_path"],
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
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        beta=cfg["beta"],
        target_noise_std=cfg.get("target_noise_std", 0.2),
        noise_clip=cfg.get("noise_clip", 0.5),
        chunk_size=chunk_size,
        actor_delay=cfg["actor_delay"],
        grad_clip=cfg["grad_clip"],
    )

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

    # Load garment list for random sampling
    parts = cfg["garment_name"].split("_")
    garment_type = f"{parts[0]}_{parts[1]}"
    garment_list_path = Path(cfg["garment_cfg_base_path"]) / "Release" / garment_type / f"{garment_type}.txt"
    with open(garment_list_path) as f:
        all_garments = [line.strip() for line in f if line.strip()]
    seen_garments = [g for g in all_garments if "Seen" in g]
    print(f"  Garment pool: {len(seen_garments)} seen + {len(all_garments) - len(seen_garments)} unseen = {len(all_garments)} total")

    # ── Phase 4: Warmup (via chunk_runner) ──
    print("\n" + "=" * 60)
    print(f"Phase 4: Warmup ({cfg['warmup_episodes']} episodes) [DECOUPLED DEBUG]")
    print("=" * 60)

    # Create MoE policy for action generation (same as eval pipeline)
    from scripts.eval_policy.moe_smolvla_policy import MoESmolVLAPolicy
    moe_policy = MoESmolVLAPolicy(device=str(device))
    moe_policy.eval()
    print(f"  MoE policy loaded for decoupled warmup")

    warmup_policy = DecoupledWarmupPolicy(
        moe_policy, vla_hook, stage1, normalizer, device, chunk_size
    )

    warmup_start = time.time()
    run_chunk_episodes(
        env=env,
        policy=warmup_policy,
        num_episodes=cfg["warmup_episodes"],
        cfg=cfg,
        args=args_namespace,
        replay_buffer=replay,
        garment_list=seen_garments,
    )
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

    # ── Phase 5: Online RL (via chunk_runner) ──
    print("\n" + "=" * 60)
    print(f"Phase 5: Online RL ({cfg['total_episodes']} episodes)")
    print("=" * 60)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_reward = -float("inf")

    rl_policy = RLActorPolicy(
        actor, vla_hook, stage1, normalizer, device,
        chunk_size, cfg["action_dim"],
    )

    def save_fn(ep_num, reward, success):
        nonlocal best_reward
        ckpt_path = output_dir / f"episode_{ep_num}.pt"
        torch.save({
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "episode": ep_num,
            "episode_reward": reward,
        }, ckpt_path)
        print(f"  Checkpoint: {ckpt_path}")
        if reward > best_reward:
            best_reward = reward
            torch.save(actor.state_dict(), output_dir / "best_actor.pt")
            print(f"  New best: {best_reward:.3f}")

    run_chunk_episodes(
        env=env,
        policy=rl_policy,
        num_episodes=cfg["total_episodes"],
        cfg=cfg,
        args=args_namespace,
        replay_buffer=replay,
        garment_list=all_garments,
        rl_trainer=trainer,
        save_fn=save_fn,
    )

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

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

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
