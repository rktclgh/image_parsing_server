FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    IMAGE_PARSER_VLM_MODE=resident \
    IMAGE_PARSER_MODEL_LOAD_ON_STARTUP=true \
    IMAGE_PARSER_MAX_CONCURRENT_GENERATIONS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY pyproject.toml README.md ./
RUN mkdir -p app && touch app/__init__.py
ARG INSTALL_EXTRAS=gpu
RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -n "$INSTALL_EXTRAS" ]; then \
        python -m pip install ".[${INSTALL_EXTRAS}]"; \
    else \
        python -m pip install .; \
    fi

COPY app ./app
COPY prompts ./prompts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
