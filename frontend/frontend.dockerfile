FROM python:3.11-slim

# System deps (keep minimal)
RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only what the frontend needs
RUN pip install --no-cache-dir streamlit requests pillow

# Copy the frontend code
COPY frontend/frontend.py /app/frontend.py

# Cloud Run provides PORT; default 8080 for local runs
ENV PORT=8080

EXPOSE 8080

# Use a shell so $PORT expands (JSON ENTRYPOINT does NOT expand env vars)
CMD ["sh", "-c", "python3 -m streamlit run /app/frontend.py --server.port=${PORT} --server.address=0.0.0.0"]
