#!/usr/bin/env python3
"""
多帧 Voting 验证实验

基于 test_router_quick.py 的特征提取流程，验证多帧 Voting 对 Router 准确率的提升效果。

实验设计:
  - 对每个 episode，提取前 N 帧的视觉特征
  - 将 N 帧的分类器 logits 累积求和 (等效于投票)
  - 对比 N=1,2,3,5,7,10 时的准确率变化
  - 重点关注 pant_short/pant_long 的混淆情况
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


# ── 数据配置 ────────────────────────────────────────────────
EPISODE_TO_TYPE = {
    (0, 250): ("top_short", 3),
    (250, 500): ("top_long", 2),
    (500, 750): ("pant_short", 0),
    (750, 1000): ("pant_long", 1),
}
TYPE_NAMES = ["pant_short", "pant_long", "top_long", "top_short"]

VOTING_FRAMES = [1, 2, 3, 5, 7, 10]  # 测试不同 N 的效果


def get_garment_type(episode_idx: int):
    for (start, end), (name, label) in EPISODE_TO_TYPE.items():
        if start <= episode_idx < end:
            return name, label
    raise ValueError(f"Unknown episode index: {episode_idx}")


def extract_rich_features(img_emb: torch.Tensor) -> torch.Tensor:
    """mean + std + max pooling → [1, hidden*3]，L2 normalized"""
    mean_f = img_emb.mean(dim=1)
    std_f = img_emb.std(dim=1)
    max_f = img_emb.max(dim=1).values
    rich = torch.cat([mean_f, std_f, max_f], dim=-1)
    return F.normalize(rich, p=2, dim=-1)


def preprocess_image(image: torch.Tensor, resize_hw, device) -> torch.Tensor:
    """复现 SmolVLA prepare_images 的预处理流程"""
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

    img = image.float()
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 3 and img.shape[-1] in [1, 3]:
        img = img.permute(2, 0, 1)   # HWC -> CHW
    img = img.unsqueeze(0).to(device)   # [1, C, H, W]
    if resize_hw is not None:
        img = resize_with_pad(img, *resize_hw, pad_value=0)
    img = img * 2.0 - 1.0   # [0,1] -> [-1,1]
    return img


def main():
    print("=" * 60)
    print("多帧 Voting Router 验证")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # ── 加载数据集 ───────────────────────────────────────────
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset_root = Path("Datasets/example/four_types_merged")
    dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)
    print(f"数据集: {dataset_root}  ({dataset.num_episodes} episodes)")

    # 抽样：每 4 个 episode 取 1 个
    sample_episodes = list(range(0, 1000, 4))
    ep_indices = np.array([x.item() for x in dataset.hf_dataset["episode_index"]])

    # 每个 episode 预先找到前 MAX_N 帧的 global 索引
    MAX_N = max(VOTING_FRAMES)
    print(f"\n收集每个 episode 的前 {MAX_N} 帧索引...")
    episode_frames: list[list[int]] = []   # [n_episodes, ≤MAX_N]
    labels: list[int] = []

    for ep_idx in tqdm(sample_episodes):
        matches = np.where(ep_indices == ep_idx)[0]
        if len(matches) == 0:
            continue
        frame_idxs = matches[:MAX_N].tolist()   # 最多取前 MAX_N 帧
        episode_frames.append(frame_idxs)
        _, label = get_garment_type(ep_idx)
        labels.append(label)

    labels_t = torch.tensor(labels)
    print(f"  有效 episodes: {len(episode_frames)}")
    print(f"  类别分布: {np.bincount(labels)}")

    # ── 加载 SmolVLA 模型 ────────────────────────────────────
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    print("\n加载 SmolVLA 模型...")
    config = SmolVLAConfig(device=device)
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", config=config)
    policy.eval()
    resize_hw = config.resize_imgs_with_padding
    hidden_size = policy.model.vlm_with_expert.config.text_config.hidden_size
    feat_dim = hidden_size * 3
    print(f"  特征维度: {feat_dim}  (mean+std+max pooling)")

    # ── 提取所有帧的特征 ─────────────────────────────────────
    # features_all[i] = Tensor[n_frames_i, feat_dim]  每个 episode 的多帧特征
    print(f"\n提取前 {MAX_N} 帧特征...")
    features_all: list[torch.Tensor] = []

    for ep_i, frame_idxs in enumerate(tqdm(episode_frames)):
        ep_feats = []
        for fidx in frame_idxs:
            frame = dataset[fidx]
            image = frame.get("observation.images.top_rgb")
            if image is None:
                continue
            img_tensor = preprocess_image(image, resize_hw, device)
            with torch.no_grad():
                try:
                    img_emb = policy.model.vlm_with_expert.embed_image(img_tensor)
                    if img_emb.dim() == 2:
                        img_emb = img_emb.unsqueeze(0)
                    feat = extract_rich_features(img_emb).cpu()   # [1, feat_dim]
                    ep_feats.append(feat)
                except Exception as e:
                    ep_feats.append(torch.zeros(1, feat_dim))
        if ep_feats:
            features_all.append(torch.cat(ep_feats, dim=0))   # [n_frames, feat_dim]
        else:
            features_all.append(torch.zeros(MAX_N, feat_dim))

    print(f"  完成，共 {len(features_all)} 个 episodes")

    # ── 训练分类器（用首帧特征，和 test_router_quick 保持一致）──
    n_ep = len(features_all)
    n_test = n_ep // 5
    perm = torch.randperm(n_ep)
    train_idx = perm[:-n_test].tolist()
    test_idx = perm[-n_test:].tolist()

    # 首帧特征用于训练
    train_feats = torch.cat([features_all[i][:1] for i in train_idx], dim=0)
    train_labels = labels_t[train_idx]

    print(f"\n训练分类器 (训练集 {len(train_idx)} 个 episodes)...")
    classifier = nn.Sequential(
        nn.Linear(feat_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 4),
    ).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss()
    train_ds = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    for epoch in range(50):
        classifier.train()
        total_loss = 0
        for bf, bl in train_loader:
            bf, bl = bf.to(device), bl.to(device)
            optimizer.zero_grad()
            loss = criterion(classifier(bf), bl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            classifier.eval()
            with torch.no_grad():
                tacc = (classifier(train_feats.to(device)).argmax(-1) == train_labels.to(device)).float().mean()
            print(f"  Epoch {epoch+1:2d}: Loss={total_loss/len(train_loader):.4f}, Train Acc={tacc:.2%}")
            classifier.train()

    # ── 多帧 Voting 测试 ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("多帧 Voting 测试结果")
    print("=" * 60)
    print(f"\n{'N frames':>10} | {'Overall':>8} | {'pant_s':>7} | {'pant_l':>7} | {'top_l':>7} | {'top_s':>7}")
    print("-" * 60)

    classifier.eval()
    test_labels_np = labels_t[test_idx].numpy()

    best_acc = 0
    results_table = []

    for N in VOTING_FRAMES:
        # 累积前 N 帧的 logits（等效于 logit-level voting）
        all_preds = []
        with torch.no_grad():
            for ep_i in test_idx:
                ep_feats = features_all[ep_i]   # [n_frames, feat_dim]
                n_avail = min(N, ep_feats.shape[0])
                feats_n = ep_feats[:n_avail].to(device)   # [n, feat_dim]

                # 每帧分别得到 logits，然后求和（sum = weighted vote）
                logits = classifier(feats_n)          # [n, 4]
                aggregated = logits.sum(dim=0)        # [4]  累计投票
                pred = aggregated.argmax().item()
                all_preds.append(pred)

        preds_np = np.array(all_preds)
        overall_acc = (preds_np == test_labels_np).mean()
        cm = confusion_matrix(test_labels_np, preds_np, labels=[0, 1, 2, 3])

        per_class_acc = []
        for cls in range(4):
            mask = test_labels_np == cls
            if mask.sum() > 0:
                per_class_acc.append((preds_np[mask] == cls).mean())
            else:
                per_class_acc.append(float('nan'))

        best_acc = max(best_acc, overall_acc)
        results_table.append((N, overall_acc, per_class_acc, cm))

        print(f"  N={N:>2}      | {overall_acc:>7.1%} | "
              f"{per_class_acc[0]:>6.1%} | {per_class_acc[1]:>6.1%} | "
              f"{per_class_acc[2]:>6.1%} | {per_class_acc[3]:>6.1%}")

    # ── 最佳 N 的混淆矩阵 ───────────────────────────────────
    best_N, best_total, best_per, best_cm = max(results_table, key=lambda x: x[1])
    print(f"\n━ 最佳结果: N={best_N} 帧, 整体准确率 {best_total:.1%} ━")
    print("\n混淆矩阵 (行=真实类，列=预测类):")
    print("              pred_ps  pred_pl  pred_tl  pred_ts")
    for i, name in enumerate(TYPE_NAMES):
        print(f"  true_{name[:2]:2s}:  {best_cm[i]}")

    # ── 提升幅度 ────────────────────────────────────────────
    n1_acc = results_table[0][1]
    nbest_acc = results_table[-1][1]
    print(f"\n单帧 (N=1) 准确率: {n1_acc:.1%}")
    print(f"多帧 (N={VOTING_FRAMES[-1]}) 准确率: {nbest_acc:.1%}")
    print(f"提升幅度: +{(nbest_acc - n1_acc):.1%}")

    # 重点：pant_short 提升
    ps_n1 = results_table[0][2][0]
    ps_nbest = results_table[-1][2][0]
    print(f"\npant_short 单帧准确率: {ps_n1:.1%}")
    print(f"pant_short 多帧准确率: {ps_nbest:.1%}  (提升 +{ps_nbest - ps_n1:.1%})")

    # ── 推荐配置 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("推荐配置")
    print("=" * 60)
    for N, acc, per_cls, _ in results_table:
        ps_str = f"{per_cls[0]:.0%}" if not np.isnan(per_cls[0]) else "n/a"
        if acc >= 0.90:
            print(f"  ✅ N={N}: {acc:.1%} (pant_short={ps_str})")
        elif acc >= 0.80:
            print(f"  ⚠️  N={N}: {acc:.1%} (pant_short={ps_str})")
        else:
            print(f"  ❌ N={N}: {acc:.1%} (pant_short={ps_str})")


if __name__ == "__main__":
    main()
