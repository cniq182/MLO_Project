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

# Because build context is en_es_translation/, these files must be inside that folder
COPY pyproject.toml uv.lock ./
COPY README.md LICENSE ./

# Your code layout inside context is: src/en_es_translation/...
COPY src ./src

# Fail fast if structure is wrong
RUN test -d /app/src/en_es_translation

# Install deps
RUN uv sync

CMD ["uv", "run", "python", "-m", "en_es_translation.predict"]
