"""
Hot Dog or Not Hot Dog — GPU Training Script for Chamber
Uses MobileNetV2 pretrained on ImageNet, fine-tuned on Food-101 hot dog images.
Reports metrics to W&B for live monitoring.
"""

import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
from datasets import load_dataset

# Try W&B (optional)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Config from env
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
LR = float(os.environ.get("LEARNING_RATE", "0.001"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "chamber-hotdog")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", "hotdog-v1")

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
    print(f"🔥 GPU detected: {GPU_NAME}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
    GPU_NAME = "Apple MPS"
else:
    DEVICE = "cpu"
    GPU_NAME = "CPU"

print(f"🖥️  Device: {DEVICE} ({GPU_NAME})")

HOTDOG_LABEL = 55  # food101 index for hot_dog


class HotDogDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(unfreeze_layers=0):
    """MobileNetV2 with optional partial unfreezing for better accuracy."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Freeze all features first
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Optionally unfreeze last N layers for fine-tuning
    if unfreeze_layers > 0:
        for layer in model.features[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2),
    )
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: {trainable:,} trainable / {total:,} total")
    
    return model


def prepare_data():
    print("📦 Loading food101 dataset...")
    ds = load_dataset("food101", split="train")

    hotdog_images, hotdog_labels = [], []
    other_images, other_labels = [], []

    print("🔍 Separating hot dogs from not hot dogs...")
    for item in ds:
        if item["label"] == HOTDOG_LABEL:
            hotdog_images.append(item["image"])
            hotdog_labels.append(1)
        else:
            other_images.append(item["image"])
            other_labels.append(0)

    n_hotdogs = len(hotdog_images)
    print(f"  🌭 Hot dogs: {n_hotdogs}")

    # Use 2x negatives for harder training
    n_neg = min(n_hotdogs * 2, len(other_images))
    random.seed(42)
    indices = random.sample(range(len(other_images)), n_neg)
    other_images = [other_images[i] for i in indices]
    other_labels = [other_labels[i] for i in indices]
    print(f"  🚫 Not hot dogs: {len(other_images)}")

    all_images = hotdog_images + other_images
    all_labels = hotdog_labels + other_labels
    return all_images, all_labels


def train():
    start_time = time.time()
    
    print(f"\n🌭 Hot Dog Classifier — Chamber GPU Training")
    print(f"{'='*55}")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"  Device: {DEVICE} ({GPU_NAME})")
    print(f"{'='*55}\n")

    # Init W&B
    if HAS_WANDB and os.environ.get("WANDB_API_KEY"):
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
                "model": "MobileNetV2",
                "dataset": "food101-hotdog",
                "gpu": GPU_NAME,
                "device": DEVICE,
                "chamber_job_id": os.environ.get("CHAMBER_JOB_ID", "unknown"),
            }
        )
        print("📈 W&B logging enabled")
    else:
        print("⚠️  W&B logging disabled (no API key)")

    # Data
    images, labels = prepare_data()
    
    full_dataset_train = HotDogDataset(images, labels, transform=get_transforms(train=True))
    full_dataset_test = HotDogDataset(images, labels, transform=get_transforms(train=False))
    
    train_size = int(0.8 * len(full_dataset_train))
    test_size = len(full_dataset_train) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_indices, test_indices = random_split(range(len(full_dataset_train)), [train_size, test_size], generator=generator)
    
    train_ds = torch.utils.data.Subset(full_dataset_train, train_indices.indices)
    test_ds = torch.utils.data.Subset(full_dataset_test, test_indices.indices)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    print(f"  Train: {len(train_ds)} | Test: {len(test_ds)}")

    # Model — unfreeze last 3 conv blocks for better fine-tuning
    model = build_model(unfreeze_layers=3).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):
            batch_images = batch_images.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()

            if (batch_idx + 1) % 5 == 0:
                batch_acc = 100. * correct / total
                print(f"  [Epoch {epoch+1}/{EPOCHS}] Batch {batch_idx+1}/{len(train_loader)} "
                      f"| Loss: {loss.item():.3f} | Acc: {batch_acc:.1f}%")

        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        scheduler.step()

        # Eval
        model.eval()
        correct = 0
        total = 0
        tp = fp = tn = fn = 0
        
        with torch.no_grad():
            for batch_images, batch_labels in test_loader:
                batch_images = batch_images.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                outputs = model(batch_images)
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
                
                # Confusion matrix
                for p, l in zip(predicted, batch_labels):
                    if p == 1 and l == 1: tp += 1
                    elif p == 1 and l == 0: fp += 1
                    elif p == 0 and l == 0: tn += 1
                    else: fn += 1

        test_acc = 100. * correct / total
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        epoch_time = time.time() - epoch_start

        print(f"\n{'─'*55}")
        print(f"  Epoch {epoch+1}/{EPOCHS} Summary ({epoch_time:.1f}s)")
        print(f"  Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")
        print(f"  Loss: {avg_loss:.3f} | Precision: {precision:.1f}% | Recall: {recall:.1f}%")
        print(f"{'─'*55}\n")

        # W&B logging
        if HAS_WANDB and wandb.run:
            wandb.log({
                "epoch": epoch + 1,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_loss": avg_loss,
                "precision": precision,
                "recall": recall,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "epoch_time_s": epoch_time,
                "lr": optimizer.param_groups[0]["lr"],
            })

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "hotdog_model_best.pth")
            print(f"  💾 New best model! ({test_acc:.1f}%)")
            
            if HAS_WANDB and wandb.run:
                wandb.run.summary["best_test_acc"] = test_acc
                wandb.run.summary["best_epoch"] = epoch + 1

    total_time = time.time() - start_time
    
    print(f"\n{'='*55}")
    print(f"✅ Training complete!")
    print(f"   Best test accuracy: {best_acc:.1f}%")
    print(f"   Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"   Device: {GPU_NAME}")
    print(f"{'='*55}")

    # Save final model
    torch.save(model.state_dict(), "hotdog_model_final.pth")

    if HAS_WANDB and wandb.run:
        wandb.run.summary["total_time_s"] = total_time
        wandb.finish()
        print("📈 W&B run finished")


if __name__ == "__main__":
    train()
