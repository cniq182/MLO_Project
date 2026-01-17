# Base image with Python 3.10
FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY README.md LICENSE ./

COPY src ./src

# Sync dependencies
RUN uv sync

# Default command (can be overridden)
CMD ["uv", "run", "python", "-m", "en_es_translation.predict"]
