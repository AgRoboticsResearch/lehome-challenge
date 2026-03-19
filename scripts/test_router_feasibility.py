#!/usr/bin/env python3
"""
Router 可行性验证脚本

测试 SmolVLM 的视觉特征能否区分 4 种服装类型：
- pant_short
- pant_long
- top_long
- top_short

用法:
    # 提取特征
    python scripts/test_router_feasibility.py extract --output features.pt

    # 训练 Router
    python scripts/test_router_feasibility.py train --features features.pt

    # 评估
    python scripts/test_router_feasibility.py eval --features features.pt --checkpoint router_best.pt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# 数据集定义
# ============================================================================

GARMENT_TYPES = {
    "pant_short": 0,
    "pant_long": 1,
    "top_long": 2,
    "top_short": 3,
}

TYPE_NAMES = ["pant_short", "pant_long", "top_long", "top_short"]


class GarmentFeatureDataset(Dataset):
    """服装特征数据集"""

    def __init__(self, features_path: str):
        data = torch.load(features_path)
        self.features = data["features"]  # [N, hidden_dim]
        self.labels = data["labels"]      # [N]
        self.garment_ids = data["garment_ids"]  # [N] - 用于按 garment 划分
        self.episode_indices = data["episode_indices"]  # [N]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "feature": self.features[idx],
            "label": self.labels[idx],
            "garment_id": self.garment_ids[idx],
            "episode_idx": self.episode_indices[idx],
        }


# ============================================================================
# Router 模型定义
# ============================================================================

class GarmentRouter(nn.Module):
    """
    服装类型 Router

    从 SmolVLM 的视觉特征预测服装类型
    """

    def __init__(
        self,
        input_dim: int = 1152,  # SmolVLM hidden size
        hidden_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.1,
    ):
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
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 视觉特征 [batch, input_dim]
        Returns:
            logits: 分类 logits [batch, num_classes]
        """
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """返回预测的类别"""
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """返回概率分布"""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


# ============================================================================
# 特征提取
# ============================================================================

def extract_features(
    dataset_root: str = "Datasets/example",
    output_path: str = "router_features.pt",
    model_path: str = "lerobot/smolvla_base",
    num_episodes_per_type: int = 25,  # 每种类型采样多少 episodes
    frames_per_episode: int = 5,       # 每个 episode 采样多少帧
    device: str = "cuda",
):
    """
    从 SmolVLM 提取视觉特征

    重要: 需要加载 SmolVLM 模型
    """
    print("=" * 60)
    print("特征提取")
    print("=" * 60)

    # 延迟导入，避免不必要的依赖
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        print(f"错误: 需要安装 lerobot: {e}")
        return

    # 加载模型
    print(f"加载模型: {model_path}")
    policy = SmolVLAPolicy.from_pretrained(model_path)
    policy.eval()
    policy.to(device)

    # 获取 VLM 的 hidden size
    hidden_size = policy.model.vlm_with_expert.config.text_config.hidden_size
    print(f"VLM hidden size: {hidden_size}")

    all_features = []
    all_labels = []
    all_garment_ids = []
    all_episode_indices = []

    dataset_names = ["pant_short_merged", "pant_long_merged", "top_long_merged", "top_short_merged"]

    for dataset_name in dataset_names:
        garment_type = dataset_name.replace("_merged", "")
        label = GARMENT_TYPES[garment_type]

        dataset_path = Path(dataset_root) / dataset_name
        if not dataset_path.exists():
            print(f"跳过: {dataset_path} 不存在")
            continue

        print(f"\n处理: {dataset_name} (label={label})")

        try:
            dataset = LeRobotDataset(
                repo_id=dataset_name,
                root=Path(dataset_root),
            )
        except Exception as e:
            print(f"  加载失败: {e}")
            continue

        # 获取所有 episode 索引
        episode_indices = dataset.episode_data_index["from"].keys()
        episode_indices = list(episode_indices)[:num_episodes_per_type]

        for ep_idx in tqdm(episode_indices, desc=f"  Episodes"):
            # 获取该 episode 的帧范围
            from_idx = dataset.episode_data_index["from"][ep_idx].item()
            to_idx = dataset.episode_data_index["to"][ep_idx].item()

            # 均匀采样帧
            frame_indices = np.linspace(from_idx, to_idx - 1, frames_per_episode, dtype=int)

            for frame_idx in frame_indices:
                # 获取图像
                frame = dataset[frame_idx]

                # 获取图像张量
                # 注意: 需要根据实际数据格式调整
                image = frame.get("observation.images.top_rgb")
                if image is None:
                    # 尝试其他键
                    for key in frame.keys():
                        if "image" in key.lower():
                            image = frame[key]
                            break

                if image is None:
                    continue

                # 预处理图像
                if isinstance(image, torch.Tensor):
                    image = image.to(device)
                else:
                    image = torch.tensor(image, device=device)

                # 确保形状正确 [1, C, H, W]
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                if image.dtype == torch.uint8:
                    image = image.float() / 255.0

                # 提取特征
                with torch.no_grad():
                    try:
                        # 方法1: 使用图像编码器 (SigLIP + Connector)
                        # embed_image 返回 [batch, num_patches, hidden_dim]
                        img_emb = policy.model.vlm_with_expert.embed_image(image)

                        # 池化到单个向量
                        if img_emb.dim() == 3:
                            visual_features = img_emb.mean(dim=1)  # [batch, hidden_dim]
                        else:
                            visual_features = img_emb

                        all_features.append(visual_features.cpu())
                        all_labels.append(label)
                        all_garment_ids.append(ep_idx // 25)  # 假设每个 garment 25 episodes
                        all_episode_indices.append(ep_idx)

                    except Exception as e:
                        print(f"    帧 {frame_idx} 特征提取失败: {e}")
                        continue

    # 合并所有特征
    if len(all_features) == 0:
        print("错误: 没有提取到任何特征")
        return

    features = torch.cat(all_features, dim=0)
    labels = torch.tensor(all_labels)
    garment_ids = torch.tensor(all_garment_ids)
    episode_indices = torch.tensor(all_episode_indices)

    print(f"\n提取完成:")
    print(f"  总样本数: {len(features)}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  类别分布: {torch.bincount(labels)}")

    # 保存
    torch.save({
        "features": features,
        "labels": labels,
        "garment_ids": garment_ids,
        "episode_indices": episode_indices,
    }, output_path)
    print(f"保存到: {output_path}")


# ============================================================================
# 训练
# ============================================================================

def train_router(
    features_path: str,
    output_dir: str = "outputs/router",
    hidden_dim: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 50,
    device: str = "cuda",
    split_by_garment: bool = True,  # 按 garment 划分，避免数据泄露
):
    """
    训练 Router 分类器
    """
    print("=" * 60)
    print("训练 Router")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print(f"加载数据: {features_path}")
    dataset = GarmentFeatureDataset(features_path)

    print(f"  总样本数: {len(dataset)}")
    print(f"  特征维度: {dataset.features.shape[1]}")
    print(f"  类别分布: {torch.bincount(dataset.labels)}")

    # 划分数据集
    if split_by_garment:
        # 按 garment 划分，避免数据泄露
        print("\n按 garment 划分数据集...")
        unique_garments = torch.unique(dataset.garment_ids)
        n_garments = len(unique_garments)
        n_train = int(n_garments * 0.7)

        # 随机选择训练 garment
        perm = torch.randperm(n_garments)
        train_garments = unique_garments[perm[:n_train]]
        val_garments = unique_garments[perm[n_train:]]

        train_mask = torch.isin(dataset.garment_ids, train_garments)
        val_mask = torch.isin(dataset.garment_ids, val_garments)

        train_indices = torch.where(train_mask)[0]
        val_indices = torch.where(val_mask)[0]

        # 创建子集
        from torch.utils.data import Subset
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

    else:
        # 简单随机划分
        n_train = int(len(dataset) * 0.8)
        train_dataset, val_dataset = random_split(dataset, [n_train, len(dataset) - n_train])

    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 创建模型
    input_dim = dataset.features.shape[1]
    model = GarmentRouter(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=4,
        dropout=dropout,
    ).to(device)

    print(f"\n模型结构:")
    print(model)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

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
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["feature"].to(device)
                labels = batch["label"].to(device)

                logits = model(features)
                loss = F.cross_entropy(logits, labels)

                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 记录
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)

        scheduler.step()

        # 打印
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "num_classes": 4,
                    "dropout": dropout,
                },
                "epoch": epoch,
                "val_acc": val_acc,
            }, os.path.join(output_dir, "router_best.pt"))

    # 保存训练历史
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\n训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型保存到: {os.path.join(output_dir, 'router_best.pt')}")

    # 最终评估
    print("\n最终评估:")
    print(classification_report(all_labels, all_preds, target_names=TYPE_NAMES))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(cm)

    return model, history


# ============================================================================
# 评估和分析
# ============================================================================

def evaluate_router(
    features_path: str,
    checkpoint_path: str,
    output_dir: str = "outputs/router/analysis",
    device: str = "cuda",
):
    """
    详细评估 Router
    """
    print("=" * 60)
    print("评估 Router")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    dataset = GarmentFeatureDataset(features_path)

    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = GarmentRouter(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"加载模型: {checkpoint_path}")
    print(f"  训练 epoch: {checkpoint['epoch']}")
    print(f"  验证准确率: {checkpoint['val_acc']:.4f}")

    # 预测所有样本
    all_features = dataset.features.to(device)
    all_labels = dataset.labels.numpy()

    with torch.no_grad():
        logits = model(all_features)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

    # 1. 整体准确率
    accuracy = (preds == all_labels).mean()
    print(f"\n整体准确率: {accuracy:.4f}")

    # 2. 每类准确率
    print("\n每类准确率:")
    for i, name in enumerate(TYPE_NAMES):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (preds[mask] == i).mean()
            print(f"  {name}: {class_acc:.4f} ({mask.sum()} 样本)")

    # 3. 混淆矩阵
    cm = confusion_matrix(all_labels, preds)
    print("\n混淆矩阵:")
    print(cm)

    # 保存混淆矩阵图
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(TYPE_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(TYPE_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # 添加数值
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    print(f"\n混淆矩阵保存到: {os.path.join(output_dir, 'confusion_matrix.png')}")

    # 4. t-SNE 可视化
    print("\n生成 t-SNE 可视化...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1))
    features_2d = tsne.fit_transform(all_features.cpu().numpy())

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["red", "blue", "green", "orange"]
    for i, name in enumerate(TYPE_NAMES):
        mask = all_labels == i
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colors[i],
            label=name,
            alpha=0.6,
            s=20,
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Visual Features t-SNE (Ground Truth)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_ground_truth.png"), dpi=150)
    print(f"t-SNE 保存到: {os.path.join(output_dir, 'tsne_ground_truth.png')}")

    # 5. 预测结果可视化
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(TYPE_NAMES):
        mask = preds == i
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colors[i],
            label=f"{name} (pred)",
            alpha=0.6,
            s=20,
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Visual Features t-SNE (Predicted)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_predicted.png"), dpi=150)
    print(f"预测 t-SNE 保存到: {os.path.join(output_dir, 'tsne_predicted.png')}")

    # 6. 置信度分析
    print("\n置信度分析:")
    confidence = probs.max(axis=1)
    correct_mask = preds == all_labels

    print(f"  整体平均置信度: {confidence.mean():.4f}")
    print(f"  正确预测平均置信度: {confidence[correct_mask].mean():.4f}")
    print(f"  错误预测平均置信度: {confidence[~correct_mask].mean():.4f}")

    # 7. 按 garment 分析 (检查泛化)
    print("\n按 garment 分析:")
    unique_garments = np.unique(dataset.garment_ids)
    for gid in unique_garments[:5]:  # 只显示前 5 个
        mask = dataset.garment_ids.numpy() == gid
        garment_acc = (preds[mask] == all_labels[mask]).mean()
        true_label = all_labels[mask][0]
        print(f"  Garment {gid} (true={TYPE_NAMES[true_label]}): acc={garment_acc:.4f} ({mask.sum()} samples)")

    return accuracy


# ============================================================================
# 简化版特征提取 (不依赖 SmolVLM)
# ============================================================================

def extract_features_simple(
    dataset_root: str = "Datasets/example",
    output_path: str = "router_features.pt",
    num_samples_per_type: int = 500,
):
    """
    简化版特征提取 - 使用预训练的图像编码器

    不需要完整的 SmolVLM，只需要一个预训练的视觉编码器
    """
    print("=" * 60)
    print("简化版特征提取")
    print("=" * 60)

    try:
        import torchvision.transforms as T
        from transformers import AutoModel, AutoProcessor
    except ImportError as e:
        print(f"错误: 需要安装 transformers 和 torchvision: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 使用 SigLIP (和 SmolVLA 相同的视觉编码器)
    model_name = "google/siglip-base-patch16-224"
    print(f"加载视觉编码器: {model_name}")

    model = AutoModel.from_pretrained(model_name).vision_model
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    model.to(device)

    hidden_size = model.config.hidden_size
    print(f"隐藏层维度: {hidden_size}")

    all_features = []
    all_labels = []
    all_garment_ids = []
    all_episode_indices = []

    dataset_names = ["pant_short_merged", "pant_long_merged", "top_long_merged", "top_short_merged"]

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    for dataset_name in dataset_names:
        garment_type = dataset_name.replace("_merged", "")
        label = GARMENT_TYPES[garment_type]

        dataset_path = Path(dataset_root) / dataset_name / "data" / "chunk-000"
        if not dataset_path.exists():
            print(f"跳过: {dataset_path} 不存在")
            continue

        print(f"\n处理: {dataset_name} (label={label})")

        # 查找图像文件
        # 假设图像存储在 images/ 目录或 parquet 中
        # 这里需要根据实际数据格式调整

        # 尝试从 parquet 读取
        import pyarrow.parquet as pq

        parquet_files = sorted(dataset_path.glob("*.parquet"))
        if not parquet_files:
            print(f"  没有找到 parquet 文件")
            continue

        sample_count = 0
        for pf in tqdm(parquet_files, desc=f"  Files"):
            if sample_count >= num_samples_per_type:
                break

            try:
                table = pq.read_table(pf)
                df = table.to_pandas()

                # 查找图像列
                image_col = None
                for col in df.columns:
                    if "image" in col.lower():
                        image_col = col
                        break

                if image_col is None:
                    continue

                # 采样
                indices = np.random.choice(len(df), min(100, len(df)), replace=False)

                for idx in indices:
                    if sample_count >= num_samples_per_type:
                        break

                    try:
                        # 获取图像
                        img_data = df.iloc[idx][image_col]

                        # 处理不同格式的图像数据
                        if isinstance(img_data, dict) and "bytes" in img_data:
                            from PIL import Image
                            import io
                            img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                        elif isinstance(img_data, np.ndarray):
                            from PIL import Image
                            img = Image.fromarray(img_data).convert("RGB")
                        else:
                            continue

                        # 预处理
                        img_tensor = transform(img).unsqueeze(0).to(device)

                        # 提取特征
                        with torch.no_grad():
                            outputs = model(img_tensor)
                            # 使用池化后的特征
                            feature = outputs.pooler_output
                            if feature is None:
                                # 使用 last_hidden_state 的平均
                                feature = outputs.last_hidden_state.mean(dim=1)

                            all_features.append(feature.cpu())
                            all_labels.append(label)

                            # 从 episode_index 获取 garment_id
                            if "episode_index" in df.columns:
                                ep_idx = df.iloc[idx]["episode_index"]
                            else:
                                ep_idx = sample_count
                            all_garment_ids.append(ep_idx // 25)
                            all_episode_indices.append(ep_idx)

                            sample_count += 1

                    except Exception as e:
                        continue

            except Exception as e:
                print(f"  读取 {pf} 失败: {e}")
                continue

        print(f"  提取了 {sample_count} 个样本")

    if len(all_features) == 0:
        print("错误: 没有提取到任何特征")
        return

    # 合并
    features = torch.cat(all_features, dim=0)
    labels = torch.tensor(all_labels)
    garment_ids = torch.tensor(all_garment_ids)
    episode_indices = torch.tensor(all_episode_indices)

    print(f"\n提取完成:")
    print(f"  总样本数: {len(features)}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  类别分布: {torch.bincount(labels)}")

    # 保存
    torch.save({
        "features": features,
        "labels": labels,
        "garment_ids": garment_ids,
        "episode_indices": episode_indices,
    }, output_path)
    print(f"保存到: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Router 可行性验证")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # extract 命令
    extract_parser = subparsers.add_parser("extract", help="提取特征")
    extract_parser.add_argument("--dataset_root", default="Datasets/example")
    extract_parser.add_argument("--output", default="router_features.pt")
    extract_parser.add_argument("--num_episodes", type=int, default=25)
    extract_parser.add_argument("--frames_per_episode", type=int, default=5)
    extract_parser.add_argument("--simple", action="store_true", help="使用简化版特征提取")

    # train 命令
    train_parser = subparsers.add_parser("train", help="训练 Router")
    train_parser.add_argument("--features", required=True, help="特征文件路径")
    train_parser.add_argument("--output_dir", default="outputs/router")
    train_parser.add_argument("--hidden_dim", type=int, default=256)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=50)

    # eval 命令
    eval_parser = subparsers.add_parser("eval", help="评估 Router")
    eval_parser.add_argument("--features", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--output_dir", default="outputs/router/analysis")

    args = parser.parse_args()

    if args.command == "extract":
        if args.simple:
            extract_features_simple(
                dataset_root=args.dataset_root,
                output_path=args.output,
            )
        else:
            extract_features(
                dataset_root=args.dataset_root,
                output_path=args.output,
                num_episodes_per_type=args.num_episodes,
                frames_per_episode=args.frames_per_episode,
            )

    elif args.command == "train":
        train_router(
            features_path=args.features,
            output_dir=args.output_dir,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

    elif args.command == "eval":
        evaluate_router(
            features_path=args.features,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
