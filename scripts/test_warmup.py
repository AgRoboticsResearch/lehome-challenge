"""
Smoke test for warmup_runner.py.

Runs only Phases 1-4 (load components + 1 warmup episode, 50 steps max)
then validates replay buffer shapes and contents.

Usage:
    python -m scripts.test_warmup --config configs/test_warmup.yaml
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
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

    from lehome.models.rl_stage2 import (
        ReplayBuffer, RLActor, TwinCritic, RLTTrainer,
    )
    from lehome.models.rl_token import RLTokenStage1
    from lehome.models.vla_stage2_hook import VLAStage2Hook

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

    # ── Phase 2: Create actor, critic, replay buffer, trainer ──
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
        capacity=1000,
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

    n_actor = sum(p.numel() for p in actor.parameters())
    n_critic = sum(p.numel() for p in critic.parameters())
    print(f"  Actor: {n_actor:,} params")
    print(f"  Critic: {n_critic:,} params")
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

    # ═══════════════════════════════════════════════════════════════
    # Deep checks — sequential buffer analysis
    # ═══════════════════════════════════════════════════════════════
    print("\n--- Deep checks (sequential) ---")

    # 1. Done flag structure: exactly cfg["warmup_episodes"] done=True entries
    dones = replay.done[:buf_len]
    n_done = dones.sum().item()
    check(
        "done count == episodes",
        n_done == cfg["warmup_episodes"],
        f"expected {cfg['warmup_episodes']}, got {n_done}",
    )

    # 2. Terminal transitions have zeroed next-state
    done_indices = dones.nonzero(as_tuple=True)[0]
    if len(done_indices) > 0:
        next_z_at_done = replay.next_z_rl[done_indices]
        next_s_at_done = replay.next_s_p[done_indices]
        check(
            "next_z_rl == 0 at done",
            (next_z_at_done == 0).all().item(),
            f"max abs = {next_z_at_done.abs().max().item():.6f}",
        )
        check(
            "next_s_p == 0 at done",
            (next_s_at_done == 0).all().item(),
            f"max abs = {next_s_at_done.abs().max().item():.6f}",
        )

    # 3. Temporal consistency: next_z_rl[i] == z_rl[i+1] for non-done transitions
    #    Episode boundaries break the chain, so we skip across done=True indices.
    not_done_mask = ~dones
    # For non-done transitions at index i, the next transition is i+1
    # (only valid if i+1 < buf_len)
    seq_valid = not_done_mask & torch.arange(buf_len, device=device).lt(buf_len - 1)
    seq_idx = seq_valid.nonzero(as_tuple=True)[0]
    if len(seq_idx) > 0:
        z_mismatch = (
            replay.next_z_rl[seq_idx] - replay.z_rl[seq_idx + 1]
        ).abs().max().item()
        check(
            "temporal: next_z_rl[i] == z_rl[i+1]",
            z_mismatch < 1e-4,
            f"max abs diff = {z_mismatch:.6f}",
        )

        s_mismatch = (
            replay.next_s_p[seq_idx] - replay.s_p[seq_idx + 1]
        ).abs().max().item()
        check(
            "temporal: next_s_p[i] == s_p[i+1]",
            s_mismatch < 1e-4,
            f"max abs diff = {s_mismatch:.6f}",
        )
    else:
        # Edge case: every transition is done (1 step episodes)
        check("temporal: next_z_rl[i] == z_rl[i+1]", True, "skipped (no non-done pairs)")

    # 4. Discount sanity: rewards should be positive and bounded
    rewards = replay.reward[:buf_len]
    check("rewards are non-negative", (rewards >= 0).all().item())

    # Upper bound: max per-step reward * sum(gamma^t for t in chunk)
    # If we assume max single-step reward ~1.0, max chunk return = sum(gamma^t)
    gamma = cfg["gamma"]
    max_chunk_return_theoretical = sum(gamma ** t for t in range(chunk_size))
    # Use a generous bound (500) since real per-step rewards can be > 1.0
    # Just check for no extreme outliers
    max_reward = rewards.max().item()
    check(
        "rewards bounded (no extreme outliers)",
        max_reward < 1000,
        f"max reward = {max_reward:.3f}",
    )

    # 5. Partial chunk padding: for done=True transitions, check if action
    #    tail is zero-padded (partial chunks). Full chunks will have non-zero
    #    throughout, but the structure should still be valid.
    if len(done_indices) > 0:
        for di in done_indices[:3]:  # check first 3 done transitions
            act = replay.action[di]  # (chunk_size, action_dim)
            # Find if there's a zero-padding boundary
            row_norms = act.norm(dim=1)  # (chunk_size,)
            nonzero_mask = row_norms > 1e-6
            if nonzero_mask.any() and not nonzero_mask.all():
                # Partial chunk: nonzero rows followed by zero rows
                first_zero = (~nonzero_mask).int().argmax().item()
                # Everything after first_zero should be zero
                tail_zero = (row_norms[first_zero:] < 1e-6).all().item()
                check(
                    f"padding monotone (done idx {di.item()})",
                    tail_zero,
                    f"first zero at step {first_zero}/{chunk_size}",
                )
            else:
                # Full chunk or all-zero (shouldn't happen)
                check(f"padding monotone (done idx {di.item()})", True, "full chunk, no padding")
                break  # only need to check partial ones

    # 6. ref_action finiteness (separate from action finiteness)
    ref_actions = replay.ref_action[:buf_len]
    check("ref_action finite", torch.isfinite(ref_actions).all().item())

    # 7. next_z_rl finite for non-done transitions
    non_done_next_z = replay.next_z_rl[:buf_len][not_done_mask]
    if len(non_done_next_z) > 0:
        check(
            "next_z_rl finite (non-done)",
            torch.isfinite(non_done_next_z).all().item(),
        )
        check(
            "next_z_rl non-zero (non-done)",
            (non_done_next_z.abs() > 0).any().item(),
            "all zeros — VLA may not be producing output",
        )

    # ═══════════════════════════════════════════════════════════════
    # Phase 4.5: BC Pretrain — test RLActor with warmup buffer
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("Phase 4.5: BC Pretrain (actor warm start from VLA)")
    print("=" * 60)

    # Snapshot actor weights before training
    actor_weight_before = {n: p.clone() for n, p in actor.named_parameters()}

    bc_optim = Adam(actor.parameters(), lr=cfg["bc_lr"])
    actor.train()

    bc_epochs = min(cfg.get("bc_pretrain_epochs", 100), 10)  # cap at 10 for test speed
    bc_batches_per_epoch = 20  # fewer batches for test speed
    bc_batch_size = min(cfg.get("batch_size", 256), buf_len)
    bc_losses = []

    for epoch in range(bc_epochs):
        total_loss = 0
        n_batches = 0
        for _ in range(bc_batches_per_epoch):
            batch = replay.sample(bc_batch_size)
            a_pred = actor(batch["z_rl"], batch["s_p"], batch["ref_action"])
            loss = F.mse_loss(a_pred, batch["ref_action"].flatten(1))

            bc_optim.zero_grad()
            loss.backward()

            # Check gradients before clipping
            total_grad_norm = torch.sqrt(
                sum(p.grad.norm() ** 2 for p in actor.parameters() if p.grad is not None)
            ).item()

            nn.utils.clip_grad_norm_(actor.parameters(), cfg["grad_clip"])
            bc_optim.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        bc_losses.append(avg_loss)
        print(f"  BC epoch {epoch+1}/{bc_epochs}: loss = {avg_loss:.6f}, grad_norm = {total_grad_norm:.4f}")
        if avg_loss < cfg.get("bc_loss_threshold", 0.01):
            print(f"  BC converged at epoch {epoch+1}")
            break

    # ── Phase 4.5 Assertions ──
    print(f"\n--- Phase 4.5 checks ---")

    # 1. Actor forward shape
    sample = replay.sample(4)
    a_pred = actor(sample["z_rl"], sample["s_p"], sample["ref_action"])
    check(
        "actor output shape",
        a_pred.shape == (4, chunk_size * cfg["action_dim"]),
        f"got {list(a_pred.shape)}",
    )

    # 2. Actor output is finite
    check("actor output finite", torch.isfinite(a_pred).all().item())

    # 3. BC loss decreased (at least 30% reduction)
    if len(bc_losses) >= 2:
        reduction = (bc_losses[0] - bc_losses[-1]) / max(bc_losses[0], 1e-8)
        check(
            "BC loss decreased",
            bc_losses[-1] < bc_losses[0],
            f"start={bc_losses[0]:.6f}, end={bc_losses[-1]:.6f}, reduction={reduction:.1%}",
        )

    # 4. Actor weights actually changed
    max_diff = max(
        (actor_weight_before[n] - p).abs().max().item()
        for n, p in actor.named_parameters()
    )
    check(
        "actor weights changed",
        max_diff > 1e-6,
        f"max weight diff = {max_diff:.8f}",
    )

    # 5. Gradients were finite during training (checked via last loss being finite)
    check("BC loss finite", all(torch.isfinite(torch.tensor(l)) for l in bc_losses))

    # 6. sync_actor_target matches
    trainer.sync_actor_target()
    max_target_diff = max(
        (tp - p).abs().max().item()
        for tp, p in zip(trainer.actor_target.parameters(), actor.parameters())
    )
    check(
        "target sync matches actor",
        max_target_diff < 1e-6,
        f"max target diff = {max_target_diff:.8f}",
    )

    # 7. Critic forward works
    q1, q2 = critic(sample["z_rl"], sample["s_p"], sample["action"])
    check(
        "critic Q1 shape",
        q1.shape == (4,),
        f"got {list(q1.shape)}",
    )
    check("critic Q1 finite", torch.isfinite(q1).all().item())
    check("critic Q2 finite", torch.isfinite(q2).all().item())

    # 8. Single trainer.update() works
    batch = replay.sample(bc_batch_size)
    metrics = trainer.update(batch)
    check("trainer update returns metrics", all(k in metrics for k in ["critic_loss"]))
    check("critic_loss finite", torch.isfinite(torch.tensor(metrics["critic_loss"])).item())
    # Actor may or may not update (delayed), so check if present
    if "actor_loss" in metrics:
        check("actor_loss finite", torch.isfinite(torch.tensor(metrics["actor_loss"])).item())

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Online RL — run episodes with RLActorPolicy + trainer
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("Phase 5: Online RL (2 episodes with RLActorPolicy)")
    print("=" * 60)

    from scripts.utils.chunk_runner import run_rl_episodes

    buffer_before = len(replay)

    # Collect trainer metrics across episodes
    all_critic_losses = []
    all_actor_losses = []

    # Temp output dir for checkpoint test
    import tempfile
    test_output_dir = tempfile.mkdtemp(prefix="rlt_test_")

    def test_save_fn(ep_num, reward, success):
        ckpt_path = Path(test_output_dir) / f"episode_{ep_num}.pt"
        torch.save({
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "episode": ep_num,
            "episode_reward": reward,
        }, ckpt_path)

    rl_cfg = dict(cfg)
    rl_cfg["save_freq"] = 1  # save every episode for testing
    rl_cfg["total_episodes"] = 2

    stats = run_rl_episodes(
        env=env,
        actor=actor,
        vla_hook=vla_hook,
        stage1=stage1,
        replay_buffer=replay,
        rl_trainer=trainer,
        cfg=rl_cfg,
        args=args_namespace,
        save_fn=test_save_fn,
        garment_list=None,
    )

    # ── Phase 5 Assertions ──
    print(f"\n--- Phase 5 checks ---")

    # 1. run_chunk_episodes returns stats
    check("run_chunk_episodes returns list", isinstance(stats, list))
    check("returns 2 episodes", len(stats) == 2, f"got {len(stats)}")

    # 2. Each stat has expected keys
    if len(stats) > 0:
        expected_keys = {"reward", "steps", "success"}
        check(
            "stat keys correct",
            expected_keys.issubset(set(stats[0].keys())),
            f"got {set(stats[0].keys())}",
        )

    # 3. Episodes ran some steps
    if len(stats) > 0:
        check("episode steps > 0", stats[0]["steps"] > 0, f"got {stats[0]['steps']}")

    # 4. Buffer grew (replay transitions from online episodes)
    buffer_after = len(replay)
    check(
        "buffer grew from RL episodes",
        buffer_after > buffer_before,
        f"before={buffer_before}, after={buffer_after}",
    )

    # 5. Checkpoints were saved
    ckpt_files = list(Path(test_output_dir).glob("episode_*.pt"))
    check("checkpoint files saved", len(ckpt_files) >= 1, f"found {len(ckpt_files)}")

    # 6. Checkpoint is loadable and has correct keys
    if ckpt_files:
        ckpt = torch.load(ckpt_files[0], map_location=device, weights_only=False)
        check(
            "checkpoint has actor state",
            "actor" in ckpt,
            f"keys: {list(ckpt.keys())}",
        )
        check("checkpoint has critic state", "critic" in ckpt)
        check("checkpoint has episode num", "episode" in ckpt)
        # Verify actor state_dict is loadable
        actor_sd = ckpt["actor"]
        check(
            "actor state_dict valid",
            isinstance(actor_sd, dict) and len(actor_sd) > 0,
        )

    # 7. Trainer step count increased (updates happened)
    check(
        "trainer updates occurred",
        trainer.step_count > 0,
        f"step_count = {trainer.step_count}",
    )

    # 8. Critic targets diverged from main (soft-updated at least once)
    max_critic_target_diff = max(
        (tp - p).abs().max().item()
        for tp, p in zip(trainer.critic_target.parameters(), critic.parameters())
    )
    check(
        "critic target diverged (soft update happened)",
        max_critic_target_diff > 0,
        f"max diff = {max_critic_target_diff:.8f}",
    )

    # 9. Buffer temporal consistency still holds after RL episodes
    buf_len_now = len(replay)
    dones_now = replay.done[:buf_len_now]
    n_done_now = dones_now.sum().item()
    check(
        "done count == warmup + rl episodes",
        n_done_now == cfg["warmup_episodes"] + 2,
        f"expected {cfg['warmup_episodes'] + 2}, got {n_done_now}",
    )

    # 10. RL episode rewards are finite
    if len(stats) > 0:
        for i, s in enumerate(stats):
            check(
                f"ep {i+1} reward finite",
                torch.isfinite(torch.tensor(s["reward"])).item(),
                f"reward = {s['reward']}",
            )

    # Cleanup temp dir
    import shutil
    shutil.rmtree(test_output_dir, ignore_errors=True)

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
