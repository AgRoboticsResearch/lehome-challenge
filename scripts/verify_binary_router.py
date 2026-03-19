#!/usr/bin/env python3
"""
验证二分类 Router：pant vs top

将 4 类合并为 2 类：
- pant_short (0) + pant_long (1) → pant (0)
- top_long (2) + top_short (3) → top (1)

用法:
    python scripts/verify_binary_router.py --features router_features.pt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm


# 原始 4 类标签
ORIGINAL_LABELS = {
    "pant_short": 0,
    "pant_long": 1,
    "top_long": 2,
    "top_short": 3,
}

# 合并后的 2 类标签
BINARY_LABELS = {
    "pant": 0,  # pant_short + pant_long
    "top": 1,   # top_long + top_short
}

# 映射：4类 → 2类
LABEL_MAP = {
    0: 0,  # pant_short → pant
    1: 0,  # pant_long → pant
    2: 1,  # top_long → top
    3: 1,  # top_short → top
}


class BinaryGarmentDataset(torch.utils.data.Dataset):
    """二分类数据集"""

    def __init__(self, features_path: str):
        data = torch.load(features_path)

        # 原始数据
        self.features = data["features"]
        self.original_labels = data["labels"]
        self.garment_ids = data["garment_ids"]

        # 转换为二分类标签
        self.binary_labels = torch.tensor(
            [LABEL_MAP[l.item()] for l in self.original_labels]
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "feature": self.features[idx],
            "label": self.binary_labels[idx],
            "original_label": self.original_labels[idx],
            "garment_id": self.garment_ids[idx],
        }


class BinaryRouter(nn.Module):
    """二分类 Router"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 二分类
        )

    def forward(self, x):
        return self.classifier(x)


def train_binary_router(
    features_path: str,
    output_dir: str = "outputs/router_binary",
    hidden_dim: int = 256,
    epochs: int = 50,
    device: str = "cuda",
):
    """训练二分类 Router"""
    print("=" * 60)
    print("训练二分类 Router: pant vs top")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    dataset = BinaryGarmentDataset(features_path)

    print(f"\n数据统计:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  特征维度: {dataset.features.shape[1]}")
    print(f"  pant 样本: {(dataset.binary_labels == 0).sum().item()}")
    print(f"  top 样本: {(dataset.binary_labels == 1).sum().item()}")

    # 打印原始 4 类分布
    print(f"\n原始 4 类分布:")
    for name, orig_label in ORIGINAL_LABELS.items():
        count = (dataset.original_labels == orig_label).sum().item()
        binary_label = LABEL_MAP[orig_label]
        binary_name = "pant" if binary_label == 0 else "top"
        print(f"  {name} ({orig_label}) → {binary_name}: {count}")

    # 按 garment 划分
    unique_garments = torch.unique(dataset.garment_ids)
    n_garments = len(unique_garments)
    n_train = int(n_garments * 0.7)

    perm = torch.randperm(n_garments)
    train_garments = unique_garments[perm[:n_train]]
    val_garments = unique_garments[perm[n_train:]]

    train_mask = torch.isin(dataset.garment_ids, train_garments)
    val_mask = torch.isin(dataset.garment_ids, val_garments)

    train_indices = torch.where(train_mask)[0]
    val_indices = torch.where(val_mask)[0]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"\n数据划分 (按 garment):")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")

    # 创建模型
    input_dim = dataset.features.shape[1]
    model = BinaryRouter(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练
    best_val_acc = 0.0

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        all_original_labels = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["feature"].to(device)
                labels = batch["label"].to(device)

                logits = model(features)
                preds = logits.argmax(dim=-1)

                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_original_labels.extend(batch["original_label"].numpy())

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "val_acc": val_acc,
            }, os.path.join(output_dir, "binary_router_best.pt"))

    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.4f}")

    # 详细分析
    print("\n" + "=" * 60)
    print("详细分析")
    print("=" * 60)

    # 1. 整体报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=["pant", "top"]))

    # 2. 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(cm)

    # 3. 按原始 4 类分析
    print("\n按原始 4 类分析:")
    all_preds = np.array(all_preds)
    all_original_labels = np.array(all_original_labels)

    for orig_name, orig_label in ORIGINAL_LABELS.items():
        mask = all_original_labels == orig_label
        if mask.sum() > 0:
            # 预期标签
            expected = LABEL_MAP[orig_label]
            # 实际正确率
            acc = (all_preds[mask] == expected).mean()
            print(f"  {orig_name}: {acc:.4f} ({mask.sum()} 样本, 预期={expected})")

    # 4. 保存混淆矩阵图
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pant", "top"])
    ax.set_yticklabels(["pant", "top"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Binary Router Confusion Matrix (Acc={best_val_acc:.4f})")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "binary_confusion_matrix.png"), dpi=150)
    print(f"\n混淆矩阵保存到: {os.path.join(output_dir, 'binary_confusion_matrix.png')}")

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="验证二分类 Router")
    parser.add_argument("--features", type=str, required=True, help="特征文件路径")
    parser.add_argument("--output_dir", default="outputs/router_binary")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    train_binary_router(
        features_path=args.features,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
