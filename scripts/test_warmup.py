"""
Smoke test for warmup_runner.py.

Runs only Phases 1-4 (load components + 1 warmup episode, 50 steps max)
then validates replay buffer shapes and contents.

Usage:
    python -m scripts.test_warmup --config configs/test_warmup.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
import os
import logging

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLEL", "1")
logging.getLogger("draccus").setLevel(logging.WARNING)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def test_warmup(cfg: dict, simulation_app):
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    device = torch.device(cfg["device"])
    chunk_size = cfg["chunk_size"]

    # ── Phase 1: Load frozen components ──
    print("=" * 60)
    print("Phase 1: Loading Frozen Components")
    print("=" * 60)

    from lehome.models.rl_stage2 import ReplayBuffer, SimpleNormalizer
    from lehome.models.rl_token import RLTokenStage1
    from lehome.models.vla_stage2_hook import VLAStage2Hook

    normalizer = SimpleNormalizer(cfg["dataset_stats_path"], device)
    print(f"  Normalizer: OK")

    vla_hook = VLAStage2Hook(
        pretrained_path=cfg["smolvla_pretrained_path"],
        device=str(device),
        task_description=cfg.get("task_description", "fold the garment"),
        dataset_stats_path=cfg["dataset_stats_path"],
    )
    print(f"  VLA hook: OK")

    stage1 = RLTokenStage1()
    ckpt = torch.load(cfg["rl_token_stage1_path"], map_location=device, weights_only=False)
    stage1.load_state_dict(ckpt["model_state_dict"])
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad = False
    stage1.to(device)
    print(f"  Stage 1: OK")

    # ── Phase 2: Create replay buffer only (no actor/critic needed) ──
    print("\n" + "=" * 60)
    print("Phase 2: Creating Replay Buffer")
    print("=" * 60)

    replay = ReplayBuffer(
        capacity=1000,
        z_rl_dim=cfg["z_rl_dim"],
        state_dim=cfg["state_dim"],
        chunk_size=chunk_size,
        action_dim=cfg["action_dim"],
        device=device,
    )
    print(f"  ReplayBuffer: capacity=1000, chunk_size={chunk_size}")

    # ── Phase 3: Create environment ──
    print("\n" + "=" * 60)
    print("Phase 3: Creating Isaac Sim Environment")
    print("=" * 60)

    env_device = cfg.get("env_device", "cpu")
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
    print(f"  Env: OK ({cfg['garment_name']})")

    # ── Phase 4: Warmup (the actual test) ──
    print("\n" + "=" * 60)
    print(f"Phase 4: Warmup ({cfg['warmup_episodes']} episode, {cfg['max_episode_steps']} steps max)")
    print("=" * 60)

    from scripts.eval_policy.moe_smolvla_policy import MoESmolVLAPolicy
    from scripts.utils.warmup_runner import run_warmup_episodes

    moe_policy = MoESmolVLAPolicy(device=str(device))
    print(f"  MoE policy: OK")

    run_warmup_episodes(
        env=env,
        moe_policy=moe_policy,
        vla_hook=vla_hook,
        stage1=stage1,
        normalizer=normalizer,
        replay_buffer=replay,
        cfg=cfg,
        args=args_namespace,
        garment_list=None,  # single garment, no switching
    )

    # ── Assertions ──
    print("\n" + "=" * 60)
    print("Validating replay buffer")
    print("=" * 60)

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  [{PASS}] {name}")
            passed += 1
        else:
            print(f"  [{FAIL}] {name}  {detail}")
            failed += 1

    buf_len = len(replay)
    check("buffer has transitions", buf_len > 0, f"got {buf_len}")

    # Expected: episodes * steps_per_episode / chunk_size
    max_chunks = cfg["warmup_episodes"] * (cfg["max_episode_steps"] // chunk_size + 1)
    check(
        "buffer size reasonable",
        0 < buf_len <= max_chunks,
        f"expected <= {max_chunks}, got {buf_len}",
    )

    if buf_len > 0:
        # Sample a transition and validate shapes
        sample = replay.sample(min(buf_len, 4))

        z_rl_shape = sample["z_rl"].shape
        check(
            "z_rl shape",
            z_rl_shape == (min(buf_len, 4), cfg["z_rl_dim"]),
            f"got {list(z_rl_shape)}",
        )

        s_p_shape = sample["s_p"].shape
        check(
            "s_p shape",
            s_p_shape == (min(buf_len, 4), cfg["state_dim"]),
            f"got {list(s_p_shape)}",
        )

        ref_shape = sample["ref_action"].shape
        check(
            "ref_action shape",
            ref_shape[0] == min(buf_len, 4) and ref_shape[2] == cfg["action_dim"],
            f"got {list(ref_shape)}",
        )

        action_shape = sample["action"].shape
        check(
            "action shape",
            action_shape == (min(buf_len, 4), chunk_size, cfg["action_dim"]),
            f"got {list(action_shape)}",
        )

        # Values should be finite (no NaN/Inf)
        check("z_rl finite", torch.isfinite(sample["z_rl"]).all().item())
        check("s_p finite", torch.isfinite(sample["s_p"]).all().item())
        check("action finite", torch.isfinite(sample["action"]).all().item())
        check("reward finite", torch.isfinite(sample["reward"]).all().item())

        # Done flag should be True for the last transition
        done_flags = sample["done"]
        check("done flag exists", done_flags is not None)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    total = passed + failed
    if failed == 0:
        print(f"All {total} checks passed!")
        print("=" * 60)
    else:
        print(f"{passed}/{total} passed, {failed} FAILED")
        print("=" * 60)
        sys.exit(1)


def main():
    import multiprocessing
    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Smoke test for warmup_runner")
    parser.add_argument("--config", type=str, default="configs/test_warmup.yaml")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Test config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import lehome.tasks.bedroom  # noqa: F401
        test_warmup(cfg, simulation_app)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
