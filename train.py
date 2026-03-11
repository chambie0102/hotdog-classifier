#!/usr/bin/env python3
"""Hot Dog vs Not Hot Dog classifier - Fine-tune ResNet-50 on Food-101."""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np

# Hyperparameters from env vars (easy to tweak between iterations)
LR = float(os.environ.get("LR", "1e-4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
CLASS_WEIGHT = float(os.environ.get("CLASS_WEIGHT", "3.0"))  # Weight for hot_dog class
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
DATA_DIR = os.environ.get("DATA_DIR", "/data")
MODEL_SAVE_PATH = os.environ.get("MODEL_SAVE_PATH", "/output/hotdog_model.pth")

# Food-101 hot_dog class index
HOTDOG_CLASS_IDX = None  # Will be determined from dataset


def get_transforms():
    """Training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def make_binary_dataset(dataset):
    """Convert Food-101 multi-class to binary: hot_dog (1) vs not_hot_dog (0)."""
    hotdog_idx = dataset.class_to_idx.get("hot_dog")
    if hotdog_idx is None:
        print("ERROR: 'hot_dog' class not found in dataset!")
        print(f"Available classes: {sorted(dataset.class_to_idx.keys())}")
        sys.exit(1)

    binary_targets = []
    for _, label in dataset.samples:
        binary_targets.append(1 if label == hotdog_idx else 0)

    dataset.targets = binary_targets
    # Override __getitem__ to return binary label
    original_samples = dataset.samples
    dataset.samples = [(path, 1 if label == hotdog_idx else 0) for path, label in original_samples]
    return dataset


def subsample_negatives(dataset, ratio=5):
    """Subsample negatives to reduce imbalance. Keep all positives + ratio*positives negatives."""
    pos_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]
    neg_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]

    n_neg = min(len(neg_indices), len(pos_indices) * ratio)
    np.random.seed(42)
    neg_sampled = list(np.random.choice(neg_indices, n_neg, replace=False))

    indices = sorted(pos_indices + neg_sampled)
    return Subset(dataset, indices)


def build_model(dropout=0.3):
    """ResNet-50 with pretrained weights, binary classification head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze early layers, fine-tune later layers
    for name, param in model.named_parameters():
        if "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout / 2),
        nn.Linear(256, 2),
    )
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch+1} [{batch_idx}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.1f}%")

    return running_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device):
    """Evaluate and return per-class metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Per-class accuracy
    classes = ["not_hot_dog", "hot_dog"]
    results = {}
    for cls_idx, cls_name in enumerate(classes):
        mask = all_labels == cls_idx
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == cls_idx).mean() * 100
        else:
            cls_acc = 0.0
        results[cls_name] = cls_acc

    overall_acc = (all_preds == all_labels).mean() * 100
    results["overall"] = overall_acc
    return results


def main():
    print("=" * 60)
    print("🌭 HOT DOG vs NOT HOT DOG CLASSIFIER")
    print("=" * 60)
    print(f"Config: LR={LR}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, "
          f"DROPOUT={DROPOUT}, CLASS_WEIGHT={CLASS_WEIGHT}")
    print(f"Device: ", end="")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU — {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("CPU (no GPU available!)")

    # Data
    print("\n📦 Loading Food-101 dataset...")
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.Food101(root=DATA_DIR, split="train", download=True, transform=train_transform)
    val_dataset = datasets.Food101(root=DATA_DIR, split="test", download=True, transform=val_transform)

    # Convert to binary
    train_dataset = make_binary_dataset(train_dataset)
    val_dataset = make_binary_dataset(val_dataset)

    # Subsample negatives for training (keep all for validation)
    train_subset = subsample_negatives(train_dataset, ratio=5)

    # Count classes
    train_pos = sum(1 for _, l in train_dataset.samples if l == 1)
    train_neg = sum(1 for _, l in train_dataset.samples if l == 0)
    print(f"Full train set: {train_pos} hot_dog, {train_neg} not_hot_dog")

    if hasattr(train_subset, 'indices'):
        sub_labels = [train_dataset.samples[i][1] for i in train_subset.indices]
        sub_pos = sum(sub_labels)
        sub_neg = len(sub_labels) - sub_pos
        print(f"Subsampled train: {sub_pos} hot_dog, {sub_neg} not_hot_dog")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    print("\n🏗️ Building model (ResNet-50 + binary head)...")
    model = build_model(dropout=DROPOUT).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    # Loss with class weights
    weights = torch.tensor([1.0, CLASS_WEIGHT]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Train
    print("\n🚀 Training...")
    best_overall = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_results = evaluate(model, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{EPOCHS} ({epoch_time:.0f}s) — LR: {current_lr:.2e}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val — hot_dog: {val_results['hot_dog']:.1f}% | "
              f"not_hot_dog: {val_results['not_hot_dog']:.1f}% | "
              f"overall: {val_results['overall']:.1f}%")

        # Save best model
        if val_results['overall'] > best_overall:
            best_overall = val_results['overall']
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  💾 Saved best model (overall: {best_overall:.1f}%)")

    total_time = time.time() - start_time

    # Final evaluation
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)

    # Load best model for final eval
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    final_results = evaluate(model, val_loader, device)

    print(f"  hot_dog accuracy:     {final_results['hot_dog']:.2f}%")
    print(f"  not_hot_dog accuracy: {final_results['not_hot_dog']:.2f}%")
    print(f"  overall accuracy:     {final_results['overall']:.2f}%")
    print(f"  training time:        {total_time/60:.1f} minutes")
    print(f"  epochs:               {EPOCHS}")
    print(f"  best model saved to:  {MODEL_SAVE_PATH}")

    # Write results JSON for easy parsing
    results_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "hot_dog_accuracy": final_results["hot_dog"],
            "not_hot_dog_accuracy": final_results["not_hot_dog"],
            "overall_accuracy": final_results["overall"],
            "training_time_minutes": total_time / 60,
            "epochs": EPOCHS,
            "hyperparameters": {
                "lr": LR, "batch_size": BATCH_SIZE, "dropout": DROPOUT,
                "class_weight": CLASS_WEIGHT,
            }
        }, f, indent=2)

    # Target check
    target = 90.0
    hotdog_pass = final_results["hot_dog"] >= target
    nothotdog_pass = final_results["not_hot_dog"] >= target
    if hotdog_pass and nothotdog_pass:
        print(f"\n✅ TARGET MET! Both classes ≥ {target}%")
    else:
        failed = []
        if not hotdog_pass:
            failed.append(f"hot_dog ({final_results['hot_dog']:.1f}%)")
        if not nothotdog_pass:
            failed.append(f"not_hot_dog ({final_results['not_hot_dog']:.1f}%)")
        print(f"\n❌ TARGET NOT MET — below {target}%: {', '.join(failed)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
