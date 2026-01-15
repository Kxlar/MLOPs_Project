# Dockerfile for model evaluation on a dataset

# Base image (we can reuse the uv image with Python 3.12)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install build tools (if needed)
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy essential project files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY models/ models/
COPY data/ data/ 

# Install project dependencies
RUN uv sync --locked --no-cache --no-install-project

# Set the entrypoint for evaluation
ENTRYPOINT ["uv", "run", "--", "python", "src/anomaly_detection/evaluate.py"]
