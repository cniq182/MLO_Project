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
COPY README.md LICENSE ./

# Source is en_es_translation/src/...
COPY src/ ./src/

RUN test -d /app/src/en_es_translation

RUN uv sync

CMD ["uv", "run", "python", "-m", "en_es_translation.predict"]
