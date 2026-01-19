# 1. Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 2. Install system build tools
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory
WORKDIR /app

# 4. Copy dependency files first (for Docker caching efficiency)
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# 5. Install dependencies
RUN uv sync --locked --no-cache --no-install-project --no-dev

# 6. Copy the actual code and models
COPY src/ src/
COPY models/ models/


# 7. Expose the port the API runs on
EXPOSE 8000

# 8. Define the command to run the API
CMD ["uv", "run", "uvicorn", "src.anomaly_detection.api:app", "--host", "0.0.0.0", "--port", "8000"]