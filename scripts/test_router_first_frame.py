#!/usr/bin/env python3
"""
首帧 Router 可行性验证

目标: 验证 SmolVLM 视觉特征能否从首帧区分 4 种服装类型

用法:
    python scripts/test_router_first_frame.py --dataset_root Datasets/example/four_types_merged

输出:
    - 首帧分类准确率
    - 混淆矩阵
    - 特征可视化 (t-SNE)
"""

import argparse
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


# ============================================================================
# 常量定义
# ============================================================================

# Episode index 到 garment type 的映射 (根据数据集合并顺序)
# Episode 0-249: top_short (250)
# Episode 250-499: top_long (250)
# Episode 500-749: pant_short (250)
# Episode 750-999: pant_long (250)

EPISODE_TO_TYPE = {
    (0, 250): ("top_short", 3),
    (250, 500): ("top_long", 2),
    (500, 750): ("pant_short", 0),
    (750, 1000): ("pant_long", 1),
}

TYPE_NAMES = ["pant_short", "pant_long", "top_long", "top_short"]


def get_garment_type(episode_idx: int) -> tuple:
    """根据 episode index 获取 garment type"""
    for (start, end), (name, label) in EPISODE_TO_TYPE.items():
        if start <= episode_idx < end:
            return name, label
    raise ValueError(f"Unknown episode index: {episode_idx}")


# ============================================================================
# 特征提取
# ============================================================================

def extract_first_frame_features(
    dataset_root: str,
    output_path: str = "first_frame_features.pt",
    max_episodes: int = None,
):
    """
    提取每个 episode 首帧的视觉特征

    使用 SmolVLA 的视觉编码器 (SigLIP)
    """
    print("=" * 60)
    print("提取首帧特征")
    print("=" * 60)

    dataset_root = Path(dataset_root)

    # 加载 LeRobot 数据集
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("错误: 需要安装 lerobot")
        return None

    print(f"加载数据集: {dataset_root}")
    dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)

    print(f"  总 episodes: {dataset.num_episodes}")
    print(f"  总 frames: {dataset.num_frames}")

    # 获取 episode_index 列
    ep_indices = dataset.hf_dataset["episode_index"]

    # 限制 episodes
    num_episodes = min(dataset.num_episodes, max_episodes) if max_episodes else dataset.num_episodes
    episode_indices = list(range(num_episodes))

    # 找到每个 episode 的首帧索引
    print("  查找首帧索引...")
    first_frame_indices = []
    labels = []

    for ep_idx in episode_indices:
        # 找到第一个匹配的帧
        for frame_idx in range(len(dataset)):
            if ep_indices[frame_idx].item() == ep_idx:
                first_frame_indices.append(frame_idx)
                break

        # 获取标签
        _, label = get_garment_type(ep_idx)
        labels.append(label)

    print(f"  首帧数量: {len(first_frame_indices)}")
    print(f"  类别分布: {np.bincount(labels)}")

    # 加载 SmolVLA 模型 (只用于特征提取)
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError:
        print("错误: 需要安装 lerobot")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n加载 SmolVLA 模型 (设备: {device})...")

    model_path = "lerobot/smolvla_base"
    policy = SmolVLAPolicy.from_pretrained(model_path)
    policy.eval()
    policy.to(device)

    # 获取 hidden size
    hidden_size = policy.model.vlm_with_expert.config.text_config.hidden_size
    print(f"  VLM hidden size: {hidden_size}")

    # 提取特征
    all_features = []

    print("\n提取特征...")
    for i, frame_idx in enumerate(tqdm(first_frame_indices)):
        # 获取帧数据
        frame = dataset[frame_idx]

        # 获取图像 (top_rgb)
        image = frame.get("observation.images.top_rgb")
        if image is None:
            print(f"  警告: Episode {i} 没有图像")
            continue

        # 预处理
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
                # 使用 SmolVLA 的图像编码器
                img_emb = policy.model.vlm_with_expert.embed_image(image)

                # 池化到单个向量
                if img_emb.dim() == 3:
                    visual_features = img_emb.mean(dim=1)
                else:
                    visual_features = img_emb

                all_features.append(visual_features.cpu())

            except Exception as e:
                print(f"  Episode {i} 特征提取失败: {e}")
                # 添加零向量作为占位符
                all_features.append(torch.zeros(1, hidden_size))

    # 合并
    features = torch.cat(all_features, dim=0)
    labels_tensor = torch.tensor(labels[:len(features)])

    print(f"\n提取完成:")
    print(f"  特征形状: {features.shape}")
    print(f"  类别分布: {torch.bincount(labels_tensor)}")

    # 保存
    torch.save({
        "features": features,
        "labels": labels_tensor,
        "episode_indices": torch.tensor(episode_indices[:len(features)]),
    }, output_path)
    print(f"保存到: {output_path}")

    return features, labels_tensor


# ============================================================================
# 分类器训练
# ============================================================================

class SimpleRouter(nn.Module):
    """简单的 Router 分类器"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def train_and_evaluate(
    features: torch.Tensor,
    labels: torch.Tensor,
    test_ratio: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    """
    训练并评估 Router

    使用简单的 train/test 划分
    """
    print("\n" + "=" * 60)
    print("训练 Router 分类器")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 划分数据集
    n_samples = len(features)
    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_test

    # 随机打乱
    perm = torch.randperm(n_samples)
    train_indices = perm[:n_train]
    test_indices = perm[n_train:]

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    print(f"训练集: {len(train_features)} 样本")
    print(f"测试集: {len(test_features)} 样本")

    # 创建 DataLoader
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    input_dim = features.shape[1]
    model = SimpleRouter(input_dim=input_dim, hidden_dim=256, num_classes=4).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 训练
    print("\n训练中...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)

        train_acc = correct / total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={train_acc:.4f}")

    # 测试
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        test_features_dev = test_features.to(device)
        test_labels_dev = test_labels.to(device)

        logits = model(test_features_dev)
        preds = logits.argmax(dim=-1)

        # 整体准确率
        accuracy = (preds == test_labels_dev).float().mean().item()
        print(f"\n整体准确率: {accuracy:.4f}")

        # 每类准确率
        print("\n每类准确率:")
        for i, name in enumerate(TYPE_NAMES):
            mask = test_labels == i
            if mask.sum() > 0:
                preds_mask = preds.cpu()[mask]
                class_acc = (preds_mask == i).float().mean().item()
                print(f"  {name}: {class_acc:.4f} ({mask.sum()} 样本)")

        # 混淆矩阵
        cm = confusion_matrix(test_labels.numpy(), preds.cpu().numpy())
        print("\n混淆矩阵:")
        print("              pred_ps  pred_pl  pred_tl  pred_ts")
        for i, name in enumerate(TYPE_NAMES):
            print(f"  true_{name[:2]:2s}:  {cm[i]}")

        # 详细报告
        print("\n分类报告:")
        print(classification_report(test_labels.numpy(), preds.cpu().numpy(), target_names=TYPE_NAMES))

    return model, accuracy


def visualize_features(features: torch.Tensor, labels: torch.Tensor, output_path: str = "router_tsne.png"):
    """t-SNE 可视化"""
    print("\n生成 t-SNE 可视化...")

    # 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features.numpy())

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["red", "blue", "green", "orange"]

    for i, name in enumerate(TYPE_NAMES):
        mask = labels == i
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colors[i],
            label=name,
            alpha=0.6,
            s=50,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("First Frame Visual Features (SmolVLM)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"保存到: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="首帧 Router 可行性验证")
    parser.add_argument("--dataset_root", default="Datasets/example/four_types_merged")
    parser.add_argument("--output", default="first_frame_features.pt")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--skip_extract", action="store_true", help="跳过特征提取，使用已有文件")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    # 特征提取
    if args.skip_extract and os.path.exists(args.output):
        print(f"加载已有特征: {args.output}")
        data = torch.load(args.output)
        features = data["features"]
        labels = data["labels"]
    else:
        result = extract_first_frame_features(
            dataset_root=args.dataset_root,
            output_path=args.output,
            max_episodes=args.max_episodes,
        )
        if result is None:
            return
        features, labels = result

    # 训练和评估
    model, accuracy = train_and_evaluate(
        features=features,
        labels=labels,
        epochs=args.epochs,
    )

    # 可视化
    visualize_features(features, labels, "router_tsne.png")

    # 结论
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    if accuracy > 0.9:
        print("✅ Router 可行性验证通过!")
        print(f"   首帧分类准确率 {accuracy:.2%} > 90%")
        print("   视觉特征可以区分 4 种服装类型")
    elif accuracy > 0.7:
        print("⚠️  Router 可行性一般")
        print(f"   首帧分类准确率 {accuracy:.2%}")
        print("   可能需要改进特征提取或分类器")
    else:
        print("❌ Router 可行性验证失败")
        print(f"   首帧分类准确率 {accuracy:.2%} < 70%")
        print("   视觉特征难以区分服装类型，需要重新设计")


if __name__ == "__main__":
    main()
