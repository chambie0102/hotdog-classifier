FROM chambie0102/chamber-demo-training:v2

WORKDIR /workspace

# Install datasets (only new dep needed)
RUN pip install --no-cache-dir datasets tqdm

COPY train_gpu.py train.py

CMD ["python", "train.py"]
