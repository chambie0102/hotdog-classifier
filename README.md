# 🌭 Hot Dog or Not Hot Dog

A binary image classifier using MobileNetV2 transfer learning. Trained on Food-101 dataset via Chamber GPU infrastructure.

## Local Training
```bash
pip install torch torchvision datasets tqdm
python train.py
```

## GPU Training (Chamber)
Push a tag to trigger Docker build + push via GitHub Actions:
```bash
git tag v3 && git push origin v3
```

Then submit to Chamber:
```bash
chamber-agent jobs submit --team <team-id> --name hotdog-v3 --gpu-type Tesla-T4 --gpus 1 --manifest hotdog-job.yaml
```

## Inference
```bash
python predict.py <image_path_or_url>
```

## Results
- **Local (Mac Mini MPS):** 89.0% test accuracy, 5 epochs, 12 seconds
- **GPU (Chamber T4):** Pending...
