#!/usr/bin/env python3
"""
训练 Garment Router

用于 MoE-SmolVLA 系统，根据 VLM 视觉特征路由到合适的 Expert。
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

# Correct episode-to-type mapping based on actual dataset structure
EPISODE_TO_TYPE = {
    (0, 250): "top_short",      # Episodes 0-249 (10 variants × 25 episodes)
    (250, 500): "top_long",      # Episodes 250-499
    (500, 750): "pant_short",    # Episodes 500-749
    (750, 1000): "pant_long",    # Episodes 750-999
}

TYPE_NAMES = ["top_short", "top_long", "pant_short", "pant_long"]
TYPE_TO_LABEL = {name: i for i, name in enumerate(TYPE_NAMES)}
NUM_CLASSES = 4


def get_garment_type(episode_idx: int) -> tuple[str, int]:
    """获取 episode 对应的 garment type 和 label

    Args:
        episode_idx: Episode index

    Returns:
        (garment_type_name, label)
    """
    for (start, end), type_name in EPISODE_TO_TYPE.items():
        if start <= episode_idx < end:
            label = TYPE_TO_LABEL[type_name]
            return type_name, label
    raise ValueError(f"Unknown episode index: {episode_idx}")


# ============================================================================
# Feature Extractor
# ============================================================================

class VLMFeatureExtractor:
    """从 SmolVLA 提取视觉特征用于路由"""

    def __init__(self, model_path: str = "lerobot/smolvla_base", device: str = "cuda"):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        self.device = torch.device(device)
        print(f"[VLMFeatureExtractor] Loading SmolVLA from {model_path}")

        config = SmolVLAConfig(device=device)
        self.policy = SmolVLAPolicy.from_pretrained(model_path, config=config)
        self.policy.eval()

        # Verify device
        vlm_device = next(self.policy.model.vlm_with_expert.vlm.parameters()).device
        print(f"[VLMFeatureExtractor] VLM device: {vlm_device}")

        self.hidden_size = self.policy.model.vlm_with_expert.config.text_config.hidden_size
        print(f"[VLMFeatureExtractor] VLM hidden size: {self.hidden_size}")
        print(f"[VLMFeatureExtractor] Rich feature size (mean+std+max): {self.hidden_size * 3}")

        # Image preprocessing
        self.resize_hw = config.resize_imgs_with_padding
        print(f"[VLMFeatureExtractor] Image preprocessing: resize_with_pad to {self.resize_hw}, normalize [0,1]->[-1,1]")

    @torch.no_grad()
    def extract_features(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        """从单张图像提取路由特征

        Args:
            image: RGB image, [H, W, C] or [C, H, W], uint8 or float [0, 1]

        Returns:
            features: [1, hidden_size * 3] - L2 normalized rich features
        """
        # Convert to tensor
        if isinstance(image, torch.Tensor):
            img_tensor = image.float()
        else:
            img_tensor = torch.from_numpy(np.array(image)).float()

        # Normalize to [0, 1] if needed
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

        # HWC -> CHW if needed
        if img_tensor.ndim == 3 and img_tensor.shape[-1] in [1, 3]:
            img_tensor = img_tensor.permute(2, 0, 1)

        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # [1, C, H, W]

        # SmolVLA preprocessing
        if self.resize_hw is not None:
            from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
            img_tensor = resize_with_pad(img_tensor, *self.resize_hw, pad_value=0)
        img_tensor = img_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        # Extract VLM embeddings
        img_emb = self.policy.model.vlm_with_expert.embed_image(img_tensor)
        if img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(0)

        # Rich features: mean + std + max pooling
        mean_f = img_emb.mean(dim=1)   # [1, D]
        std_f = img_emb.std(dim=1)    # [1, D]
        max_f = img_emb.max(dim=1).values  # [1, D]

        rich = torch.cat([mean_f, std_f, max_f], dim=-1)  # [1, D*3]
        rich = F.normalize(rich, p=2, dim=-1)  # L2 normalize

        return rich


# ============================================================================
# Router Classifier
# ============================================================================

class GarmentRouter(nn.Module):
    """Garment 类型分类器 (Router)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.7),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim]

        Returns:
            logits: [batch, num_classes]
        """
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> dict:
        """预测并返回详细信息

        Args:
            x: [batch, input_dim]

        Returns:
            dict with:
                predicted_class: [batch] - predicted class indices
                class_name: str - name of predicted class
                probabilities: [batch, num_classes] - softmax probabilities
                confidence: [batch] - max probability
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        predicted_class = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values

        return {
            "logits": logits,
            "predicted_class": predicted_class,
            "probabilities": probs,
            "confidence": confidence,
        }


# ============================================================================
# Dataset Preparation
# ============================================================================

def prepare_router_dataset(
    dataset_root: Path,
    extractor: VLMFeatureExtractor,
    episode_stride: int = 4,
    cache_dir: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """准备 Router 训练数据集

    Args:
        dataset_root: Dataset root path
        extractor: VLM feature extractor
        episode_stride: Stride for sampling episodes (default: 4, 每4个取1个)
        cache_dir: Cache directory for extracted features

    Returns:
        (features, labels): features [N, D], labels [N]
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"\n[Dataset] Loading dataset from {dataset_root}")
    dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)

    print(f"[Dataset] Total episodes: {dataset.num_episodes}")
    print(f"[Dataset] Total frames: {dataset.num_frames}")

    # Sample episodes
    test_episodes = list(range(0, 1000, episode_stride))
    print(f"[Dataset] Sampling {len(test_episodes)} episodes (stride={episode_stride})")

    # Find first frames
    ep_indices = np.array([x.item() for x in dataset.hf_dataset["episode_index"]])

    first_frame_indices = []
    labels = []

    print("[Dataset] Finding first frames...")
    for ep_idx in tqdm(test_episodes):
        matches = np.where(ep_indices == ep_idx)[0]
        if len(matches) > 0:
            first_frame_indices.append(matches[0])
            _, label = get_garment_type(ep_idx)
            labels.append(label)

    print(f"[Dataset] Found {len(first_frame_indices)} first frames")
    label_counts = np.bincount(labels)
    print(f"[Dataset] Label distribution: {dict(zip(TYPE_NAMES, label_counts))}")

    # Check cache
    if cache_dir is not None:
        cache_file = cache_dir / "router_features.pt"
        if cache_file.exists():
            print(f"[Dataset] Loading cached features from {cache_file}")
            cached = torch.load(cache_file)
            return cached["features"], cached["labels"]

    # Extract features
    cam_key = "observation.images.top_rgb"
    print(f"[Dataset] Extracting features from {cam_key}...")

    all_features = []
    for i, frame_idx in enumerate(tqdm(first_frame_indices, desc="Extracting features")):
        frame = dataset[frame_idx]
        image = frame.get(cam_key)

        if image is None:
            print(f"[Dataset] Warning: No image for frame {frame_idx}, using zeros")
            feat_dim = extractor.hidden_size * 3
            all_features.append(torch.zeros(1, feat_dim))
            continue

        # Extract features
        feat = extractor.extract_features(image)  # [1, D*3]
        all_features.append(feat.cpu())

    features = torch.cat(all_features, dim=0)
    labels = torch.tensor(labels[:len(features)])

    print(f"[Dataset] Features shape: {features.shape}")
    print(f"[Dataset] Labels shape: {labels.shape}")

    # Cache features
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "labels": labels}, cache_file)
        print(f"[Dataset] Cached features to {cache_file}")

    return features, labels


# ============================================================================
# Training
# ============================================================================

def train_router(
    features: torch.Tensor,
    labels: torch.Tensor,
    output_dir: Path,
    device: str = "cuda",
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 5e-4,
    val_split: float = 0.2,
):
    """训练 Router

    Args:
        features: [N, D] training features
        labels: [N] training labels
        output_dir: Output directory for checkpoints
        device: Device to use
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        val_split: Validation split ratio
    """
    device = torch.device(device)

    # Split train/val
    n_samples = len(features)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    # Stratified split
    train_features = []
    val_features = []
    train_labels = []
    val_labels = []

    for class_idx in range(NUM_CLASSES):
        mask = labels == class_idx
        class_feats = features[mask]
        class_labels = labels[mask]

        n_class_val = int(len(class_feats) * val_split)
        perm = torch.randperm(len(class_feats))

        val_features.append(class_feats[perm[:n_class_val]])
        val_labels.append(class_labels[perm[:n_class_val]])
        train_features.append(class_feats[perm[n_class_val:]])
        train_labels.append(class_labels[perm[n_class_val:]])

    train_features = torch.cat(train_features, dim=0)
    val_features = torch.cat(val_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    print(f"\n[Train] Train set: {len(train_features)} samples")
    print(f"[Train] Val set: {len(val_features)} samples")

    # Create model
    input_dim = features.shape[1]
    model = GarmentRouter(input_dim=input_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Data loaders
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    best_val_acc = 0.0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Train] Starting training for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_f, batch_l in train_loader:
            batch_f, batch_l = batch_f.to(device), batch_l.to(device)

            optimizer.zero_grad()
            logits = model(batch_f)
            loss = criterion(logits, batch_l)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Validation
        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                train_preds = model(train_features.to(device)).argmax(-1)
                train_acc = (train_preds == train_labels.to(device)).float().mean().item()

                val_preds = model(val_features.to(device)).argmax(-1)
                val_acc = (val_preds == val_labels.to(device)).float().mean().item()

            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Loss={total_loss/len(train_loader):.4f}, "
                  f"Train Acc={train_acc:.4f}, "
                  f"Val Acc={val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_dir = output_dir / "checkpoints" / "best"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(model, optimizer, scheduler, checkpoint_dir, val_acc, epoch)

    # Save final model
    checkpoint_dir = output_dir / "checkpoints" / "last"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, scheduler, checkpoint_dir, val_acc, epochs - 1)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_dir: Path,
    val_acc: float,
    epoch: int,
):
    """保存模型检查点"""
    checkpoint = {
        "router_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": {
            "input_dim": model.classifier[0].in_features,
            "hidden_dims": [
                layer.out_features
                for i, layer in enumerate(model.classifier)
                if isinstance(layer, nn.Linear) and i < len(model.classifier) - 1
            ],
            "num_classes": NUM_CLASSES,
            "type_names": TYPE_NAMES,
            "episode_to_type": EPISODE_TO_TYPE,
        },
        "metrics": {
            "val_acc": val_acc,
            "epoch": epoch,
        },
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(checkpoint, checkpoint_dir / "router.pt")
    print(f"[Checkpoint] Saved to {checkpoint_dir / 'router.pt'}")


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_router(model: GarmentRouter, features: torch.Tensor, labels: torch.Tensor, device: str = "cuda"):
    """评估 Router 性能"""
    device = torch.device(device)
    model.eval()

    with torch.no_grad():
        predictions = model(features.to(device)).argmax(-1).cpu()

    accuracy = (predictions == labels).float().mean().item()
    cm = confusion_matrix(labels.numpy(), predictions.numpy())

    print("\n" + "=" * 60)
    print("Router Evaluation")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print("              " + "  ".join([f"pred_{name[:7]:7s}" for name in TYPE_NAMES]))
    for i, name in enumerate(TYPE_NAMES):
        print(f"  true_{name[:7]:7s}: {cm[i]}")

    print("\nPer-class Accuracy:")
    for i in range(NUM_CLASSES):
        class_mask = labels == i
        if class_mask.sum() > 0:
            class_acc = (predictions[class_mask] == i).float().mean().item()
            print(f"  {TYPE_NAMES[i]:15s}: {class_acc:.4f} ({class_mask.sum():3d} samples)")

    return accuracy, cm


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Garment Router for MoE-SmolVLA")
    parser.add_argument("--dataset_root", type=str, default="Datasets/example/four_types_merged",
                        help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="outputs/train/router",
                        help="Output directory for checkpoints")
    parser.add_argument("--vlm_model", type=str, default="lerobot/smolvla_base",
                        help="VLM model path for feature extraction")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--episode_stride", type=int, default=4,
                        help="Stride for sampling episodes (default: 4, 每4个取1个)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for extracted features (default: output_dir/cache)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode: train or eval")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for eval mode")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Garment Router Training")
    print("=" * 60)

    # Setup cache dir
    if args.cache_dir is None:
        cache_dir = Path(args.output_dir) / "cache"
    else:
        cache_dir = Path(args.cache_dir)

    # Initialize feature extractor
    extractor = VLMFeatureExtractor(model_path=args.vlm_model, device=args.device)

    if args.mode == "train":
        # Prepare dataset
        features, labels = prepare_router_dataset(
            dataset_root=Path(args.dataset_root),
            extractor=extractor,
            episode_stride=args.episode_stride,
            cache_dir=cache_dir,
        )

        # Train
        model = train_router(
            features=features,
            labels=labels,
            output_dir=Path(args.output_dir),
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_split=args.val_split,
        )

        # Final evaluation
        print("\n" + "=" * 60)
        print("Final Evaluation on Training Data")
        print("=" * 60)
        evaluate_router(model, features, labels, device=args.device)

    elif args.mode == "eval":
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for eval mode")

        # Load checkpoint
        print(f"\n[Eval] Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)

        # Rebuild model
        config = checkpoint["config"]
        model = GarmentRouter(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            num_classes=config["num_classes"],
        ).to(args.device)
        model.load_state_dict(checkpoint["router_state_dict"])
        model.eval()

        print(f"[Eval] Loaded model from epoch {checkpoint['metrics']['epoch']}")
        print(f"[Eval] Validation accuracy: {checkpoint['metrics']['val_acc']:.4f}")

        # Load test data
        features, labels = prepare_router_dataset(
            dataset_root=Path(args.dataset_root),
            extractor=extractor,
            episode_stride=args.episode_stride,
            cache_dir=cache_dir,
        )

        # Evaluate
        evaluate_router(model, features, labels, device=args.device)


if __name__ == "__main__":
    main()
