# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Python and build tools installation
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY data/ data/
COPY models/ models/

RUN uv sync --locked --no-cache --no-install-project

ENTRYPOINT ["uv", "run", "src/anomaly_detection/train.py"]