FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
  && python -m pip install --no-cache-dir uv \
  && uv --version

# These files must exist inside en_es_translation/
COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY en_es_translation/LICENSE ./

# Copy the entire en_es_translation directory to maintain the package structure
# pyproject.toml expects: en_es_translation/src/en_es_translation
COPY en_es_translation/ ./en_es_translation/

RUN test -d /app/en_es_translation/src/en_es_translation

RUN uv sync

CMD ["uv", "run", "python", "-m", "en_es_translation.predict"]
