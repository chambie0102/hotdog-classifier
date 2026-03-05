"""
Hot Dog or Not Hot Dog — Inference
Pass an image path or URL and get a verdict.
"""

import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO

MODEL_PATH = "hotdog_model.pth"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ["🚫 NOT Hot Dog", "🌭 HOT DOG!"]


def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 2),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def load_image(path_or_url):
    if path_or_url.startswith("http"):
        response = requests.get(path_or_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(path_or_url).convert("RGB")
    return img


def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = load_model()
    img = load_image(image_path)
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    label = LABELS[predicted.item()]
    conf = confidence.item() * 100

    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"  Confidence: {conf:.1f}%")
    print(f"{'='*40}\n")

    return predicted.item(), conf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path_or_url>")
        sys.exit(1)
    predict(sys.argv[1])
