FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt --no-deps
COPY . .
RUN pip install -e .
RUN mkdir -p /app/data /app/demo_run /app/checkpoints
ENV PYTHONUNBUFFERED=1
CMD ["python", "adaptive_varnet_model/train_adaptive_varnet_demo.py", \
     "--data_path", "/app/data", \
     "--challenge", "multicoil", \
     "--gpus", "0", \
     "--num_workers", "0", \
     "--batch_size", "1", \
     "--max_epochs", "1", \
     "--limit_train_batches", "1", \
     "--default_root_dir", "./demo_run", \
     "--wandb", "False"]
