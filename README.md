# 🌭 Hot Dog or Not Hot Dog

AI image classifier trained on Chamber's GPU infrastructure. Inspired by Silicon Valley.

## Architecture
- **Model:** MobileNetV2 (pretrained on ImageNet, fine-tuned on Food-101)
- **Dataset:** Food-101 hot_dog class vs random other food classes
- **Training:** GPU-accelerated via Chamber K8s cluster
- **Metrics:** W&B logging (project: chamber-hotdog)

## Quick Start

### Build & Push (via GitHub Actions)
```bash
git tag v1 && git push origin v1
# Or trigger manually via Actions tab
```

### Submit to Chamber GPU
```bash
kubectl apply -f job.yaml
kubectl logs -f job/hotdog-classifier-v1
```

### Local Inference
```bash
pip install -r requirements.txt
python predict.py <image_path_or_url>
```

## Training Script
- `docker/train_gpu.py` — GPU training with W&B logging, MobileNetV2 transfer learning
- `docker/Dockerfile` — CUDA 12.4 + PyTorch container
- `train.py` — Local training (MPS/CPU)
- `predict.py` — Inference script

## Results
Training on T4 GPU via Chamber cluster. Metrics tracked on W&B.
