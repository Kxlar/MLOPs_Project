# Dockerfile for end-to-end data drift demo (generate drifted datasets, run API, call API, plot histograms)
# NOTE: This runs the experiment at container runtime (ENTRYPOINT), not during image build.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update \
    && apt install --no-install-recommends -y build-essential gcc \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY scripts/ scripts/

# Install dependencies (skip installing the local project as a package)
RUN uv sync --locked --no-cache --no-install-project --no-dev

RUN chmod +x scripts/run_data_drift_demo.sh

EXPOSE 8000

ENTRYPOINT ["bash", "scripts/run_data_drift_demo.sh"]
