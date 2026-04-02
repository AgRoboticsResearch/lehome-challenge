"""
RL Token Stage 1: Offline training of Encoder/Decoder.

Usage:
    # Phase 1: Precompute prefix embeddings (one-time, slow)
    python -m scripts.train_rl_token_stage1 --config configs/train_rl_token.yaml --precompute

    # Phase 2: Train RL Token Encoder/Decoder (fast, no SmolVLA needed)
    python -m scripts.train_rl_token_stage1 --config configs/train_rl_token.yaml --train

    # Or do both phases in sequence:
    python -m scripts.train_rl_token_stage1 --config configs/train_rl_token.yaml --precompute --train
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "lehome"))
sys.path.insert(0, str(PROJECT_ROOT / "submission" / "source_code" / "lerobot_policies_smolvla"))

from lehome.models.rl_token import RLTokenStage1
from lehome.models.vla_prefix_hook import (
    VLAPrefixHook,
    PrefixEmbeddingDataset,
    precompute_prefix_embeddings,
)


def load_config(config_path: str) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def warmup_lr_lambda(step: int, warmup_steps: int):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return 1.0


def train(cfg: dict):
    device = torch.device(cfg.get("device", "cpu"))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix_cache_path = Path(cfg["prefix_cache_path"])
    if not prefix_cache_path.exists():
        print(f"Prefix cache not found at {prefix_cache_path}")
        print("Run with --precompute first.")
        return

    # Memory-mapped binary format: avoids loading ~30GB into RAM
    import numpy as np

    mmap_path = prefix_cache_path.with_suffix(".mmap.bin")
    meta_path = prefix_cache_path.with_suffix(".meta.pt")

    if meta_path.exists():
        meta = torch.load(meta_path, weights_only=True)
        num_frames = meta["num_frames"]
        total_tokens = meta["total_tokens"]
        d_model = meta["d_model"]
        arr = np.memmap(str(mmap_path), dtype="float16", mode="r",
                        shape=(num_frames, total_tokens, d_model))
        cache_tensor = torch.from_numpy(arr)
        print(f"Loaded {num_frames} frames from {mmap_path} (memory-mapped)")
    else:
        # One-time conversion: dict -> binary memmap (minimal RAM)
        print(f"Loading prefix cache from {prefix_cache_path} ...")
        cache = torch.load(prefix_cache_path, map_location="cpu", mmap=True, weights_only=True)
        indices = sorted(cache.keys())
        num_frames = len(indices)
        print(f"  Loaded {num_frames} frames (tensors memory-mapped)")

        sample = cache[indices[0]]
        total_tokens = sample.shape[0]
        d_model = sample.shape[1]
        print(f"  Sample shape: [{total_tokens}, {d_model}]")

        print("  Converting to memory-mapped binary format ...")
        arr = np.memmap(str(mmap_path), dtype="float16", mode="w+",
                        shape=(num_frames, total_tokens, d_model))
        for i, idx in enumerate(indices):
            arr[i] = cache[idx].numpy()
        arr.flush()
        del cache, arr

        torch.save({"num_frames": num_frames, "total_tokens": total_tokens, "d_model": d_model}, meta_path)
        print(f"  Saved to {mmap_path} + {meta_path}")

        arr = np.memmap(str(mmap_path), dtype="float16", mode="r",
                        shape=(num_frames, total_tokens, d_model))
        cache_tensor = torch.from_numpy(arr)

    num_img_tokens = cfg.get("num_image_tokens", 192)
    num_lang_tokens = cfg.get("num_lang_tokens", 3)
    num_state_tokens = total_tokens - num_img_tokens - num_lang_tokens
    print(f"  Token layout: {num_img_tokens} image + {num_lang_tokens} lang + {num_state_tokens} state = {total_tokens}")

    dataset = PrefixEmbeddingDataset(cache_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=PrefixEmbeddingDataset.collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    model = RLTokenStage1(
        d_model=cfg.get("d_model", 960),
        nhead=cfg.get("num_heads", 15),
        dim_feedforward=cfg.get("dim_feedforward", 1920),
        encoder_layers=cfg.get("encoder_layers", 2),
        decoder_layers=cfg.get("decoder_layers", 2),
        num_image_tokens=num_img_tokens,
        num_state_tokens=num_state_tokens,
        num_lang_tokens=num_lang_tokens,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  Encoder layers: {cfg.get('encoder_layers', 2)}")
    print(f"  Decoder layers: {cfg.get('decoder_layers', 2)}")
    print(f"  keep_mask: {model.keep_mask.sum().item()}/{model.keep_mask.shape[0]} tokens kept")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_lr_lambda(step, cfg.get("warmup_steps", 200)),
    )

    torch.manual_seed(cfg.get("seed", 42))

    log_path = output_dir / "training_log.jsonl"
    best_loss = float("inf")
    global_step = 0
    data_iter = iter(dataloader)

    print(f"\nStarting training for {cfg['steps']} steps on {device} ...")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Learning rate: {cfg.get('lr', 1e-4)}")
    print(f"  Log every {cfg.get('log_freq', 100)} steps, save every {cfg.get('save_freq', 1000)} steps")
    print()

    model.train()
    start_time = time.time()

    for step in range(cfg["steps"]):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(device=device, dtype=torch.float32)

        result = model(batch)
        loss = result["loss"]

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        optimizer.step()
        scheduler.step()

        global_step += 1

        if (step + 1) % cfg.get("log_freq", 100) == 0:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            tokens_per_frame = result["z_target"].shape[1]
            msg = (
                f"[Step {step+1}/{cfg['steps']}] "
                f"loss={loss.item():.6f} "
                f"grad_norm={grad_norm:.4f} "
                f"lr={lr:.2e} "
                f"z_rl_norm={result['z_rl'].norm(dim=-1).mean().item():.4f} "
                f"tokens_kept={tokens_per_frame} "
                f"time={elapsed:.1f}s"
            )
            print(msg)

            log_entry = {
                "step": step + 1,
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "lr": lr,
                "z_rl_norm": result["z_rl"].norm(dim=-1).mean().item(),
                "elapsed_s": elapsed,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        if (step + 1) % cfg.get("save_freq", 1000) == 0 or step == cfg["steps"] - 1:
            is_best = loss.item() < best_loss
            if is_best:
                best_loss = loss.item()

            ckpt_dir = output_dir / "checkpoints" / f"step_{step+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "step": step + 1,
                    "loss": loss.item(),
                    "config": cfg,
                },
                ckpt_dir / "rl_token_stage1.pt",
            )
            torch.save(model.encoder.state_dict(), ckpt_dir / "encoder.pt")
            print(f"  Saved checkpoint to {ckpt_dir}" + (" (best)" if is_best else ""))

            if is_best:
                best_dir = output_dir / "checkpoints" / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "step": step + 1,
                        "loss": loss.item(),
                        "config": cfg,
                    },
                    best_dir / "rl_token_stage1.pt",
                )
                torch.save(model.encoder.state_dict(), best_dir / "encoder.pt")

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to {output_dir / 'checkpoints'}")
    print(f"Training log: {log_path}")


def precompute(cfg: dict):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = Path(cfg["dataset_root"])
    print(f"Loading dataset from {dataset_root} ...")
    dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)
    print(f"  {len(dataset)} frames, {dataset.num_episodes} episodes")

    print("Initializing VLAPrefixHook ...")
    hook = VLAPrefixHook(
        pretrained_path=cfg.get("smolvla_pretrained_path"),
        device=cfg.get("device", "cpu"),
        task_description=cfg.get("task_description", "fold the garment"),
    )

    print(f"  Language tokens: {hook.num_lang_tokens}")
    print(f"  Image tokens per camera: {hook.num_image_tokens}")
    print(f"  State tokens: {hook.num_state_tokens}")

    precompute_prefix_embeddings(
        hook=hook,
        lerobot_dataset=dataset,
        output_path=cfg["prefix_cache_path"],
        batch_size=1,
    )


def main():
    parser = argparse.ArgumentParser(description="RL Token Stage 1 Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--precompute", action="store_true", help="Precompute prefix embeddings")
    parser.add_argument("--train", action="store_true", help="Train RL Token Encoder/Decoder")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config: {json.dumps(cfg, indent=2)}")

    if args.precompute:
        precompute(cfg)

    if args.train:
        train(cfg)

    if not args.precompute and not args.train:
        print("Specify --precompute and/or --train")


if __name__ == "__main__":
    main()
