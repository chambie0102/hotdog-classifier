# 🌭 Hot Dog vs Not Hot Dog Classifier

Binary image classifier: **hot dog** or **not hot dog**. Fine-tunes ResNet-50 on Food-101.

## Architecture
- **Base:** ResNet-50 (ImageNet V2 pretrained)
- **Head:** Dropout → 256-d → ReLU → Dropout → 2-class output
- **Frozen:** Layers 1-2 (fine-tune layers 3-4 + head)
- **Dataset:** Food-101 (hot_dog class vs subsampled others)

## Hyperparameters (via env vars)
| Var | Default | Description |
|-----|---------|-------------|
| `LR` | `1e-4` | Learning rate |
| `BATCH_SIZE` | `64` | Batch size |
| `EPOCHS` | `10` | Training epochs |
| `DROPOUT` | `0.3` | Dropout rate |
| `CLASS_WEIGHT` | `3.0` | Weight for hot_dog class |

## Run on Chamber
```bash
chamber-agent -y agent workloads submit \
  --name hotdog-v1 \
  --team-id 56512b13-e1f7-4dbb-a3bd-8122fe54a90c \
  --gpu-type Tesla-T4 \
  --manifest @job.yaml
```

## CI/CD
Push a tag (`v1`, `v2`, etc.) → GitHub Actions builds and pushes to Docker Hub.
