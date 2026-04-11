FROM huggingface/lerobot-gpu:latest

WORKDIR /app

# The pre-built image has a venv at /lerobot/.venv with all lerobot extras installed.
# Inherit the venv so our policy.py can import from lerobot.
ENV PATH=/lerobot/.venv/bin:$PATH

COPY dummy_docker_policy/server.py dummy_docker_policy/policy.py ./
RUN mkdir -p /app/checkpoints /app/datasets

EXPOSE 8080

# Configuration via environment variables (override at runtime)
ENV CHECKPOINT_DIR=/app/checkpoints
ENV DATASET_ROOT=/app/datasets
ENV DEVICE=cuda

CMD ["python", "policy.py"]
