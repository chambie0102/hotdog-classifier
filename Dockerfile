FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install minimal deps
RUN pip install --no-cache-dir torchvision numpy

# Copy training script
COPY train.py .

# Default command
CMD ["python", "-u", "train.py"]
