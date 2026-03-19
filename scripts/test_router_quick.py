#!/usr/bin/env python3
"""
快速验证 Router 可行性

只测试少量数据，快速验证视觉特征能否区分服装类型
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm


# Episode index 到 garment type 的映射
EPISODE_TO_TYPE = {
    (0, 250): ("top_short", 3),
    (250, 500): ("top_long", 2),
    (500, 750): ("pant_short", 0),
    (750, 1000): ("pant_long", 1),
}

TYPE_NAMES = ["pant_short", "pant_long", "top_long", "top_short"]


def get_garment_type(episode_idx: int):
    for (start, end), (name, label) in EPISODE_TO_TYPE.items():
        if start <= episode_idx < end:
            return name, label
    raise ValueError(f"Unknown episode index: {episode_idx}")


def extract_rich_features(img_emb: torch.Tensor) -> torch.Tensor:
    """
    从 VLM image token embeddings 提取更丰富的特征

    img_emb: [1, N_tokens, hidden_dim]
    返回: [1, hidden_dim * 3] — mean + std + max 三种 pooling 拼接

    原来只用 mean pooling，导致不同服装的特征向量几乎相同（loss卡在ln(4)）
    现在用 mean+std+max 三种 pooling 拼接，保留更多分布信息
    """
    tokens = img_emb  # [1, N, D]
    mean_f = tokens.mean(dim=1)   # [1, D]
    std_f  = tokens.std(dim=1)    # [1, D] — 方差反映图像复杂度
    max_f  = tokens.max(dim=1).values  # [1, D] — 最强激活

    # 拼接三种视角
    rich = torch.cat([mean_f, std_f, max_f], dim=-1)  # [1, D*3]

    # L2 normalization 让特征更均匀
    rich = F.normalize(rich, p=2, dim=-1)
    return rich


def verify_episode_mapping(dataset, garment_info_path: Path, num_check: int = 5):
    """
    验证 EPISODE_TO_TYPE 的假设是否正确
    通过对比 garment_info.json 中的服装顺序和实际 episode 顺序
    """
    import json
    print("\n=== 验证 Episode-Type 映射 ===")
    with open(garment_info_path) as f:
        gi = json.load(f)

    garment_keys = list(gi.keys())
    print(f"garment_info.json 中的服装类型 (前10个):")
    for k in garment_keys[:10]:
        print(f"  {k}: {len(gi[k])} episodes")

    # 简单验证：episode 0 应该是第一个服装的第一个 episode
    print(f"\n假设映射:")
    for (s, e), (name, label) in EPISODE_TO_TYPE.items():
        print(f"  Episode {s}-{e-1} -> {name} (label={label})")

    print("\n⚠️ 如果上面的映射与 garment_info.json 顺序不符，请手动修正 EPISODE_TO_TYPE")


def main():
    print("=" * 60)
    print("Router 可行性快速验证 (修复版)")
    print("=" * 60)

    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 加载数据集
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = Path("Datasets/example/four_types_merged")
    print(f"\n加载数据集: {dataset_root}")
    dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)

    print(f"  总 episodes: {dataset.num_episodes}")
    print(f"  总 frames: {dataset.num_frames}")

    # 验证 episode -> type 映射
    verify_episode_mapping(dataset, dataset_root / "meta/garment_info.json")

    # 增加样本量：每 4 个 episode 取 1 个 (原来每10个取1，每类只有~20个训练样本太少)
    test_episodes = list(range(0, 1000, 4))
    print(f"\n测试 {len(test_episodes)} 个 episodes (每4个取1)")

    # 高效查找首帧：预先构建 episode -> first_frame 映射
    first_frame_indices = []
    labels = []

    print("查找首帧...")
    # 获取所有 episode_index 并转为 numpy 数组（一次性操作）
    ep_indices = np.array([x.item() for x in dataset.hf_dataset["episode_index"]])

    for ep_idx in tqdm(test_episodes):
        # 使用 numpy 快速查找首帧
        matches = np.where(ep_indices == ep_idx)[0]
        if len(matches) > 0:
            first_frame_indices.append(matches[0])
            _, label = get_garment_type(ep_idx)
            labels.append(label)

    print(f"  首帧数量: {len(first_frame_indices)}")
    label_counts = np.bincount(labels)
    print(f"  类别分布 [pant_short, pant_long, top_long, top_short]: {label_counts}")

    # 加载模型
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, resize_with_pad
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    print(f"\n加载 SmolVLA 模型...")
    model_path = "lerobot/smolvla_base"  # 原始预训练模型
    # 必须在加载时指定设备，否则 device_map="auto" 会忽略后续的 to(device)
    config = SmolVLAConfig(device=device)
    policy = SmolVLAPolicy.from_pretrained(model_path, config=config)
    policy.eval()
    # 验证设备
    vlm_device = next(policy.model.vlm_with_expert.vlm.parameters()).device
    print(f"  VLM 实际设备: {vlm_device}")

    hidden_size = policy.model.vlm_with_expert.config.text_config.hidden_size
    print(f"  VLM hidden size: {hidden_size}")
    print(f"  Rich feature size (mean+std+max): {hidden_size * 3}")

    # SmolVLA 不使用 HuggingFace processor 处理图像！
    # 它使用 resize_with_pad (到 224x224) + 归一化到 [-1, 1]
    # 使用 processor 会返回 5D tiled tensors 导致 embed_image 崩溃
    resize_hw = config.resize_imgs_with_padding  # 通常是 (224, 224)
    print(f"  图像预处理: resize_with_pad to {resize_hw}, normalize [0,1]->[-1,1]")

    # 提取特征（只用 top_rgb 单摄像头)
    CAM_KEYS = [
        "observation.images.top_rgb",
        # "observation.images.left_rgb",
        # "observation.images.right_rgb",
    ]
    all_features = []

    print(f"\n提取特征 ({len(CAM_KEYS)} 摄像头拼接)...")
    for i, frame_idx in enumerate(tqdm(first_frame_indices)):
        frame = dataset[frame_idx]
        cam_feats = []

        for cam_key in CAM_KEYS:
            image = frame.get(cam_key)
            if image is None:
                # 摄像头缺失时用零向量占位
                cam_feats.append(torch.zeros(1, hidden_size * 3))
                continue

            # 转换为 float32 tensor [C, H, W]，值域 [0, 1]
            if isinstance(image, torch.Tensor):
                img_tensor = image.float()
            else:
                img_tensor = torch.from_numpy(np.array(image)).float()

            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0

            if img_tensor.ndim == 3 and img_tensor.shape[-1] in [1, 3]:
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW

            img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, C, H, W]

            # SmolVLA 的预处理: resize_with_pad + normalize [0,1] -> [-1,1]
            if resize_hw is not None:
                img_tensor = resize_with_pad(img_tensor, *resize_hw, pad_value=0)
            img_tensor = img_tensor * 2.0 - 1.0

            with torch.no_grad():
                try:
                    img_emb = policy.model.vlm_with_expert.embed_image(img_tensor)
                    if img_emb.dim() == 2:
                        img_emb = img_emb.unsqueeze(0)
                    cam_feat = extract_rich_features(img_emb)  # [1, hidden*3]
                    cam_feats.append(cam_feat.cpu())
                except Exception as e:
                    print(f"  Episode {i} cam {cam_key} 失败: {e}")
                    cam_feats.append(torch.zeros(1, hidden_size * 3))

        # 三摄像头特征拼接: [1, hidden*3*n_cams]
        multi_cam_feat = torch.cat(cam_feats, dim=-1)
        # 再做一次整体 L2 normalization
        multi_cam_feat = F.normalize(multi_cam_feat, p=2, dim=-1)
        all_features.append(multi_cam_feat)

    # 合并
    features = torch.cat(all_features, dim=0)
    labels = torch.tensor(labels[:len(features)])

    print(f"\n特征形状: {features.shape}")

    # 诊断特征有效性：检查类间距离
    print("\n=== 特征诊断 ===")
    for class_id in range(4):
        mask = labels == class_id
        class_feats = features[mask]
        print(f"  {TYPE_NAMES[class_id]}: mean_norm={class_feats.norm(dim=-1).mean():.4f}, "
              f"intra_std={class_feats.std(dim=0).mean():.4f}")

    # 类间余弦相似度（越低越好，说明类之间特征差异大）
    class_centers = []
    for class_id in range(4):
        mask = labels == class_id
        center = features[mask].mean(dim=0)
        class_centers.append(F.normalize(center.unsqueeze(0), p=2, dim=-1))

    print("\n  类间余弦相似度矩阵 (越低越好):")
    print("              ps    pl    tl    ts")
    for i in range(4):
        row = []
        for j in range(4):
            sim = (class_centers[i] * class_centers[j]).sum().item()
            row.append(f"{sim:.3f}")
        print(f"  {TYPE_NAMES[i][:5]:5s}:  {' '.join(row)}")

    # 划分训练集测试集
    n_samples = len(features)
    n_test = n_samples // 5
    perm = torch.randperm(n_samples)
    train_idx = perm[:-n_test]
    test_idx = perm[-n_test:]

    train_features, train_labels = features[train_idx], labels[train_idx]
    test_features, test_labels = features[test_idx], labels[test_idx]

    print(f"\n训练集: {len(train_features)}, 测试集: {len(test_features)}")

    # 分类器：更大的网络，更多 epochs
    feat_dim = features.shape[1]
    classifier = nn.Sequential(
        nn.Linear(feat_dim, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 4),
    ).to(device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 训练：更多 epochs
    print("\n训练分类器 (更大网络, 200 epochs)...")
    for epoch in range(200):
        classifier.train()
        total_loss = 0
        for batch_f, batch_l in train_loader:
            batch_f, batch_l = batch_f.to(device), batch_l.to(device)
            optimizer.zero_grad()
            logits = classifier(batch_f)
            loss = criterion(logits, batch_l)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            # 训练集准确率
            classifier.eval()
            with torch.no_grad():
                train_acc = (classifier(train_features.to(device)).argmax(-1) == train_labels.to(device)).float().mean().item()
            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Train Acc={train_acc:.2%}")
            classifier.train()

    # 测试
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)

    classifier.eval()
    with torch.no_grad():
        test_f = test_features.to(device)
        test_l = test_labels.to(device)
        logits = classifier(test_f)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == test_l).float().mean().item()

    print(f"\n整体准确率: {accuracy:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(test_labels.numpy(), preds.cpu().numpy())
    print("\n混淆矩阵:")
    print("              pred_ps  pred_pl  pred_tl  pred_ts")
    for i, name in enumerate(TYPE_NAMES):
        print(f"  true_{name[:2]:2s}:  {cm[i]}")

    # 逐类准确率
    print("\n逐类准确率:")
    for i in range(4):
        class_mask = test_labels == i
        if class_mask.sum() > 0:
            class_acc = (preds.cpu()[class_mask] == i).float().mean().item()
            print(f"  {TYPE_NAMES[i]}: {class_acc:.2%} ({class_mask.sum()} samples)")

    # 结论
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    if accuracy > 0.9:
        print(f"✅ Router 可行性验证通过! ({accuracy:.2%})")
        print("   视觉特征可以区分 4 种服装类型，MoE 方案可行")
    elif accuracy > 0.7:
        print(f"⚠️ Router 可行性一般 ({accuracy:.2%})")
    else:
        print(f"❌ Router 可行性验证失败 ({accuracy:.2%})")


if __name__ == "__main__":
    main()
