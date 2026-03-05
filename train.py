"""
Hot Dog or Not Hot Dog — Transfer Learning Classifier
Uses MobileNetV2 pretrained on ImageNet, fine-tuned on hot dog images.
Downloads food101 hot_dog class + random other classes for negative examples.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import random

# Config
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
NUM_WORKERS = 0
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "hotdog_model.pth"

# food101 label index for hot_dog
HOTDOG_LABEL = 55


class HotDogDataset(Dataset):
    """Binary classifier: hot_dog (1) vs everything else (0)."""

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
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model():
    """MobileNetV2 with frozen features, trainable classifier."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 2),
    )
    return model


def prepare_data():
    """Load food101, extract hot dogs vs not hot dogs (balanced)."""
    print("📦 Loading food101 dataset (this may take a minute)...")
    ds = load_dataset("food101", split="train")

    hotdog_images = []
    hotdog_labels = []
    other_images = []
    other_labels = []

    print("🔍 Separating hot dogs from not hot dogs...")
    for item in tqdm(ds, desc="Processing"):
        if item["label"] == HOTDOG_LABEL:
            hotdog_images.append(item["image"])
            hotdog_labels.append(1)
        else:
            other_images.append(item["image"])
            other_labels.append(0)

    # Balance: same number of not-hot-dogs as hot dogs
    n_hotdogs = len(hotdog_images)
    print(f"  🌭 Hot dogs: {n_hotdogs}")

    random.seed(42)
    indices = random.sample(range(len(other_images)), n_hotdogs)
    other_images = [other_images[i] for i in indices]
    other_labels = [other_labels[i] for i in indices]
    print(f"  🚫 Not hot dogs: {len(other_images)}")

    all_images = hotdog_images + other_images
    all_labels = hotdog_labels + other_labels

    return all_images, all_labels


def train():
    print(f"🌭 Hot Dog Classifier — Training on {DEVICE}")
    print("=" * 50)

    images, labels = prepare_data()

    dataset = HotDogDataset(images, labels, transform=get_transforms(train=True))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    # Use eval transforms for test set
    test_ds.dataset = HotDogDataset(images, labels, transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"  Train: {len(train_ds)} | Test: {len(test_ds)}")

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_images, batch_labels in pbar:
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
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100.*correct/total:.1f}%")

        train_acc = 100. * correct / total
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_images, batch_labels in test_loader:
                batch_images = batch_images.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                outputs = model(batch_images)
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()

        test_acc = 100. * correct / total
        print(f"  → Train: {train_acc:.1f}% | Test: {test_acc:.1f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  💾 Saved best model ({test_acc:.1f}%)")

    print(f"\n✅ Training complete! Best test accuracy: {best_acc:.1f}%")
    print(f"   Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
