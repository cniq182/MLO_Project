FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps (install once)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
  && rm -rf /var/lib/apt/lists/*

# Install uv once
RUN python -m pip install --no-cache-dir --upgrade pip \
  && python -m pip install --no-cache-dir uv \
  && uv --version

# Copy metadata first for better caching
COPY pyproject.toml uv.lock ./
COPY README.md LICENSE ./

# IMPORTANT: copy the path that egg_base expects
COPY en_es_translation ./en_es_translation

# (Optional but helpful) fail fast with a clear message
RUN test -d /app/en_es_translation/src

# Install deps
RUN uv sync

CMD ["uv", "run", "python", "-m", "en_es_translation.predict"]
