"""
RL Token Stage 1 Quality Evaluation.

Usage:
    # Full evaluation (reconstruction + linear probe + temporal):
    python scripts/eval_rl_token_stage1.py \
        --checkpoint outputs/rl_token/stage1/checkpoints/best/rl_token_stage1.pt \
        --prefix_cache outputs/rl_token/prefix_cache_top_long \
        --dataset_root Datasets/example/top_long_merged

    # Reconstruction only (no dataset needed):
    python scripts/eval_rl_token_stage1.py \
        --checkpoint outputs/rl_token/stage1/checkpoints/step_5000/rl_token_stage1.pt \
        --prefix_cache outputs/rl_token/prefix_cache_top_long
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "lehome"))

from lehome.models.rl_token import RLTokenStage1


# ─── Data Loading ────────────────────────────────────────────────


def load_prefix_cache(cache_path: str):
    """Load prefix cache, supporting both .pt and .mmap.bin formats."""
    cache_path = Path(cache_path)

    # Determine mmap and meta paths
    if cache_path.suffix == ".pt":
        mmap_path = cache_path.with_suffix(".mmap.bin")
        meta_path = cache_path.with_suffix(".meta.pt")
    elif cache_path.suffix == ".bin":
        mmap_path = cache_path
        meta_path = cache_path.parent / cache_path.name.replace(".mmap.bin", ".meta.pt")
    else:
        mmap_path = cache_path.with_suffix(".mmap.bin")
        meta_path = cache_path.with_suffix(".meta.pt")

    if mmap_path.exists() and meta_path.exists():
        meta = torch.load(meta_path, weights_only=True)
        num_frames = meta["num_frames"]
        total_tokens = meta["total_tokens"]
        d_model = meta["d_model"]
        arr = np.memmap(str(mmap_path), dtype="float16", mode="r",
                        shape=(num_frames, total_tokens, d_model))
        data = torch.from_numpy(arr)
        print(f"  Loaded mmap cache: {num_frames} frames, {total_tokens} x {d_model}")
        return data, num_frames, total_tokens, d_model

    pt_path = cache_path.with_suffix(".pt") if cache_path.suffix != ".pt" else cache_path
    if pt_path.exists():
        print(f"  Loading from .pt: {pt_path} (consider converting to mmap first)")
        cache = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
        indices = sorted(cache.keys())
        num_frames = len(indices)
        sample = cache[indices[0]]
        total_tokens, d_model = sample.shape
        data = torch.stack([cache[i].float() for i in indices], dim=0)
        return data, num_frames, total_tokens, d_model

    raise FileNotFoundError(f"Prefix cache not found at {cache_path}")


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained RLTokenStage1 from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = RLTokenStage1(
        d_model=cfg.get("d_model", 960),
        nhead=cfg.get("num_heads", 15),
        dim_feedforward=cfg.get("dim_feedforward", 1920),
        encoder_layers=cfg.get("encoder_layers", 2),
        decoder_layers=cfg.get("decoder_layers", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, ckpt


def load_dataset_info(dataset_root: str, num_frames_cap: int):
    """Load actions, states, and episode indices from LeRobot dataset.

    Tries fast parquet column read first, falls back to sequential loading.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=Path(dataset_root).name, root=Path(dataset_root))
    N = min(len(ds), num_frames_cap)

    # Fast path: try reading from parquet columns directly
    try:
        import pyarrow.parquet as pq

        parquet_dir = Path(dataset_root) / "data"
        parquet_files = sorted(parquet_dir.glob("**/*.parquet"))
        if parquet_files:
            print(f"  Fast-loading from {len(parquet_files)} parquet file(s) ...")
            table = pq.concat_tables([pq.read_table(f) for f in parquet_files])

            actions = torch.from_numpy(
                np.stack(table.column("action").to_numpy())
            ).float()[:N]
            states = torch.from_numpy(
                np.stack(table.column("observation.state").to_numpy())
            ).float()[:N]
            episode_indices = torch.from_numpy(
                np.array(table.column("episode_index").to_numpy())
            ).long()[:N]
            return actions, states, episode_indices, N
    except Exception as e:
        print(f"  Fast path failed ({e}), using sequential loading ...")

    # Slow fallback: iterate one by one
    actions = torch.zeros(N, 12)
    states = torch.zeros(N, 12)
    episode_indices = torch.zeros(N, dtype=torch.long)

    for i in tqdm(range(N), desc="  Loading dataset", ncols=80):
        frame = ds[i]
        act = frame["action"]
        if not isinstance(act, torch.Tensor):
            act = torch.from_numpy(np.asarray(act))
        actions[i] = act.float()
        s = frame.get("observation.state", frame.get("state"))
        if not isinstance(s, torch.Tensor):
            s = torch.from_numpy(np.asarray(s))
        if s.ndim > 1:
            s = s.squeeze()
        states[i] = s.float()
        ep = frame.get("episode_index", 0)
        if isinstance(ep, torch.Tensor):
            ep = ep.item()
        episode_indices[i] = int(ep)

    return actions, states, episode_indices, N


# ─── Evaluation Functions ────────────────────────────────────────


@torch.no_grad()
def eval_reconstruction(model, data, device, num_samples=2000, batch_size=256):
    """Reconstruction quality: per-token cosine similarity and MSE."""
    N = data.shape[0]
    n = min(num_samples, N)
    indices = torch.randperm(N)[:n]

    all_cos = []

    for i in range(0, n, batch_size):
        batch_idx = indices[i : i + batch_size]
        batch = data[batch_idx].to(device=device, dtype=torch.float32)
        result = model(batch)

        pred = result["pred"]        # (B, 193, 960)
        target = result["z_target"]  # (B, 193, 960)

        cos = F.cosine_similarity(pred, target, dim=-1)  # (B, 193)
        all_cos.append(cos.cpu())

    all_cos = torch.cat(all_cos, dim=0)  # (n, 193)
    per_token = all_cos.mean(dim=0)       # (193,)

    return {
        "overall_cos_sim": all_cos.mean().item(),
        "overall_cos_sim_std": all_cos.std().item(),
        "image_cos_sim": all_cos[:, :192].mean().item(),
        "state_cos_sim": all_cos[:, 192:].mean().item(),
        "per_camera": {
            "top":   all_cos[:, :64].mean().item(),
            "left":  all_cos[:, 64:128].mean().item(),
            "right": all_cos[:, 128:192].mean().item(),
        },
        "first_token_cos_sim": per_token[0].item(),
        "last_token_cos_sim": per_token[-1].item(),
        "per_token_cos_sim": per_token,
    }


@torch.no_grad()
def encode_all_zrl(model, data, device, batch_size=256):
    """Encode all frames to z_rl vectors."""
    N = data.shape[0]
    z_rls = []
    for i in tqdm(range(0, N, batch_size), desc="  Encoding z_rl", ncols=80):
        batch = data[i : i + batch_size].to(device=device, dtype=torch.float32)
        z_target = model.apply_keep_mask(batch)
        z_rl = model.encoder(z_target)
        z_rls.append(z_rl.cpu())
    return torch.cat(z_rls, dim=0)  # (N, 960)


@torch.no_grad()
def compute_z_target_mean(data, keep_mask, batch_size=256):
    """Mean-pooled z_target as baseline (same 960D as z_rl)."""
    N = data.shape[0]
    means = []
    for i in tqdm(range(0, N, batch_size), desc="  Mean pooling", ncols=80):
        batch = data[i : i + batch_size].float()
        z_target = batch[:, keep_mask, :]  # (B, 193, 960)
        means.append(z_target.mean(dim=1))  # (B, 960)
    return torch.cat(means, dim=0)  # (N, 960)


def _train_probe(X_train, y_train, X_val, y_val, device, epochs=50, lr=1e-3):
    """Train linear regression probe, return best val MSE."""
    probe = nn.Linear(X_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=256, shuffle=True
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        probe.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(probe(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_loss = criterion(probe(X_val.to(device)), y_val.to(device)).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss


def eval_linear_probe(z_rls, z_target_means, actions, states, episode_indices, N, device):
    """Linear probe: compare z_rl vs baselines for action prediction.

    Three probes with fair comparison:
      1. z_rl (960D)          — encoder output (what Stage 2 will use)
      2. z_target_mean (960D) — naive mean pooling of VLM tokens
      3. state (12D)           — raw joint positions (lower bound)
    """
    # Split by episode (avoid leakage from adjacent frames)
    unique_eps = episode_indices.unique()
    n_train_eps = int(len(unique_eps) * 0.8)
    train_eps = set(unique_eps[:n_train_eps].tolist())

    train_mask = torch.tensor(
        [episode_indices[i].item() in train_eps for i in range(N)]
    )
    val_mask = ~train_mask

    print(f"  Split: {train_mask.sum()} train / {val_mask.sum()} val "
          f"({n_train_eps}/{len(unique_eps) - n_train_eps} episodes)")

    results = {}

    # Probe 1: z_rl → action
    print("  [1/3] Training z_rl probe ...")
    results["z_rl_mse"] = _train_probe(
        z_rls[train_mask], actions[train_mask],
        z_rls[val_mask], actions[val_mask], device,
    )

    # Probe 2: z_target_mean → action
    print("  [2/3] Training z_target_mean probe ...")
    results["z_target_mean_mse"] = _train_probe(
        z_target_means[train_mask], actions[train_mask],
        z_target_means[val_mask], actions[val_mask], device,
    )

    # Probe 3: state → action
    print("  [3/3] Training state-only probe ...")
    results["state_mse"] = _train_probe(
        states[train_mask], actions[train_mask],
        states[val_mask], actions[val_mask], device,
    )

    results["improvement_vs_mean_pooling_pct"] = (
        (results["z_target_mean_mse"] - results["z_rl_mse"])
        / results["z_target_mean_mse"] * 100
    )
    results["improvement_vs_state_pct"] = (
        (results["state_mse"] - results["z_rl_mse"])
        / results["state_mse"] * 100
    )

    return results


def eval_temporal(z_rls, episode_indices, N):
    """Temporal consistency: within-episode smoothness vs cross-episode distance."""
    unique_eps = episode_indices.unique()

    adj_cos_sims = []
    for ep in unique_eps:
        ep_idx = (episode_indices == ep).nonzero(as_tuple=True)[0]
        if len(ep_idx) < 2:
            continue
        ep_z = z_rls[ep_idx]
        cos = F.cosine_similarity(ep_z[:-1], ep_z[1:], dim=-1)
        adj_cos_sims.append(cos)

    adj_cos_sims = torch.cat(adj_cos_sims)

    # Cross-episode: random pairs from different episodes
    cross_sims = []
    for _ in range(min(2000, N)):
        i, j = torch.randint(0, N, (2,))
        if episode_indices[i] != episode_indices[j]:
            cross_sims.append(
                F.cosine_similarity(z_rls[i:i+1], z_rls[j:j+1], dim=-1).item()
            )

    return {
        "adjacent_cos_sim_mean": adj_cos_sims.mean().item(),
        "adjacent_cos_sim_std": adj_cos_sims.std().item(),
        "cross_episode_cos_sim": (
            sum(cross_sims) / len(cross_sims) if cross_sims else 0.0
        ),
        "num_episodes": len(unique_eps),
    }


# ─── Main ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="RL Token Stage 1 Quality Evaluation"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to rl_token_stage1.pt checkpoint")
    parser.add_argument("--prefix_cache", required=True,
                        help="Prefix cache path (base, .pt, or .mmap.bin)")
    parser.add_argument("--dataset_root", default=None,
                        help="Dataset root for linear probe & temporal analysis")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save eval results JSON")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Samples for reconstruction eval")
    parser.add_argument("--probe_epochs", type=int, default=50,
                        help="Linear probe training epochs")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("  RL Token Stage 1 — Quality Evaluation")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────
    print(f"\nLoading model: {args.checkpoint}")
    model, cfg, ckpt = load_model(args.checkpoint, device)
    print(f"  Trained {ckpt['step']} steps, loss at save: {ckpt['loss']:.6f}")

    # ── Load prefix cache ─────────────────────────────────────
    print(f"\nLoading prefix cache: {args.prefix_cache}")
    data, num_frames, total_tokens, d_model = load_prefix_cache(args.prefix_cache)

    has_dataset = args.dataset_root is not None
    all_results = {"checkpoint": args.checkpoint, "step": ckpt["step"]}

    # ── 1. Reconstruction Quality ─────────────────────────────
    print("\n" + "─" * 50)
    print("  1. Reconstruction Quality")
    print("─" * 50)

    t0 = time.time()
    recon = eval_reconstruction(model, data, device, args.num_samples)

    print(f"  Overall cosine sim:    {recon['overall_cos_sim']:.4f} "
          f"(std {recon['overall_cos_sim_std']:.4f})")
    print(f"  Image tokens cosine:   {recon['image_cos_sim']:.4f}")
    print(f"  State token cosine:    {recon['state_cos_sim']:.4f}")
    print(f"  Per camera:")
    for cam in ("top", "left", "right"):
        print(f"    {cam:>5s}: {recon['per_camera'][cam]:.4f}")

    print(f"\n  Teacher-forcing analysis:")
    print(f"    1st token (z_rl only): {recon['first_token_cos_sim']:.4f}")
    print(f"    Last token (full ctx):  {recon['last_token_cos_sim']:.4f}")

    per_tok = recon["per_token_cos_sim"]
    sorted_tok = per_tok.sort()
    worst_v, worst_i = sorted_tok.values[:5], sorted_tok.indices[:5]
    print(f"\n  Worst 5 tokens:")
    for v, idx in zip(worst_v, worst_i):
        if idx < 64:
            loc = f"top:{idx}"
        elif idx < 128:
            loc = f"left:{idx - 64}"
        elif idx < 192:
            loc = f"right:{idx - 128}"
        else:
            loc = "state"
        print(f"    Token {idx.item():3d} ({loc:>10s}): {v.item():.4f}")

    print(f"  ({time.time()-t0:.1f}s)")

    all_results["reconstruction"] = {
        k: v for k, v in recon.items() if k != "per_token_cos_sim"
    }

    # ── 2 & 3. Linear Probe + Temporal (require dataset) ──────
    if has_dataset:
        # Load dataset once
        print("\n" + "─" * 50)
        print("  Loading dataset for linear probe & temporal ...")
        print("─" * 50)
        t_load = time.time()
        actions, states, episode_indices, N = load_dataset_info(
            args.dataset_root, num_frames
        )
        print(f"  Loaded {N} frames ({time.time()-t_load:.1f}s)")

        # Encode z_rl ONCE, reuse for both probe and temporal
        print("\n  Encoding all frames to z_rl ...")
        t_enc = time.time()
        z_rls = encode_all_zrl(model, data[:N], device)
        print(f"  Done: {z_rls.shape} ({time.time()-t_enc:.1f}s)")

        # Compute mean pooling baseline
        z_target_means = compute_z_target_mean(data[:N], model.keep_mask)

        # ── 2. Linear Probe ───────────────────────────────────
        print("\n" + "─" * 50)
        print("  2. Linear Probe (z_rl → action prediction)")
        print("─" * 50)

        t0 = time.time()
        probe = eval_linear_probe(
            z_rls, z_target_means, actions, states, episode_indices, N, device
        )
        all_results["linear_probe"] = probe

        print(f"\n  Results:")
        print(f"    z_rl (960D)          MSE: {probe['z_rl_mse']:.6f}")
        print(f"    z_target_mean (960D) MSE: {probe['z_target_mean_mse']:.6f}")
        print(f"    state (12D)          MSE: {probe['state_mse']:.6f}")
        print(f"    vs mean pooling:     {probe['improvement_vs_mean_pooling_pct']:+.1f}%")
        print(f"    vs state only:       {probe['improvement_vs_state_pct']:+.1f}%")
        print(f"  ({time.time()-t0:.1f}s)")

        # ── 3. Temporal Consistency ───────────────────────────
        print("\n" + "─" * 50)
        print("  3. Temporal Consistency")
        print("─" * 50)

        t0 = time.time()
        # Reuse z_rls already computed above
        temporal = eval_temporal(z_rls, episode_indices, N)
        all_results["temporal"] = temporal

        gap = temporal["adjacent_cos_sim_mean"] - temporal["cross_episode_cos_sim"]
        print(f"  Adjacent-frame cos sim: {temporal['adjacent_cos_sim_mean']:.4f} "
              f"(std {temporal['adjacent_cos_sim_std']:.4f})")
        print(f"  Cross-episode cos sim:  {temporal['cross_episode_cos_sim']:.4f}")
        print(f"  Gap (adj - cross):      {gap:.4f} "
              f"{'(good)' if gap > 0.05 else '(small — may not distinguish episodes)'}")
        print(f"  Episodes: {temporal['num_episodes']}")
        print(f"  ({time.time()-t0:.1f}s)")
    else:
        print("\n  [Linear probe & temporal skipped — provide --dataset_root to enable]")

    # ── Save ──────────────────────────────────────────────────
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        def serialize(o):
            if isinstance(o, torch.Tensor):
                return o.tolist()
            return o

        with open(out / "eval_results.json", "w") as f:
            json.dump(all_results, f, default=serialize, indent=2)
        print(f"\n  Results saved to {out / 'eval_results.json'}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    verdicts = []
    if recon["overall_cos_sim"] > 0.9:
        verdicts.append("Reconstruction: GOOD (cos_sim > 0.9)")
    elif recon["overall_cos_sim"] > 0.8:
        verdicts.append("Reconstruction: OK (cos_sim 0.8-0.9)")
    else:
        verdicts.append("Reconstruction: WEAK (cos_sim < 0.8)")

    if has_dataset:
        if probe["improvement_vs_mean_pooling_pct"] > 0:
            verdicts.append(
                f"Linear probe: z_rl is {probe['improvement_vs_mean_pooling_pct']:.1f}% "
                f"better than mean pooling"
            )
        else:
            verdicts.append(
                f"Linear probe: z_rl is WORSE than mean pooling "
                f"({probe['improvement_vs_mean_pooling_pct']:.1f}%)"
            )

    if has_dataset:
        if gap > 0.05:
            verdicts.append("Temporal: good episode structure separation")
        else:
            verdicts.append("Temporal: weak episode structure separation")

    for v in verdicts:
        print(f"  - {v}")

    print("=" * 60)


if __name__ == "__main__":
    main()
