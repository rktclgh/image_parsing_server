# Docker Linux GPU service

This package runs `image_parsing_server` as a one-worker FastAPI/Uvicorn service on the Linux GPU host. Keep the service to one Uvicorn worker: each worker would load its own Gemma runtime, and the target 16 GB VRAM host is sized for a single resident model copy.

## Worktree and branch

Use a branch based on `develop` so packaging work does not modify the PR #18 branch while it waits for review:

```bash
cd /home/song/oh-my-design/image_parsing_server
git fetch origin develop
git worktree add -b feat/docker-linux-service-packaging \
  /home/song/oh-my-design/image_parsing_server-docker-packaging origin/develop
cd /home/song/oh-my-design/image_parsing_server-docker-packaging
```

## Configure

```bash
cp .env.example .env
# Edit .env only on the controlled Linux host. Do not commit secrets.
```

Compose can start with its checked-in defaults when `.env` is absent, but copying `.env.example` makes host-specific ports and runtime choices explicit.

Important defaults:

- `IMAGE_PARSER_VLM_MODE=resident`
- `IMAGE_PARSER_MODEL_LOAD_ON_STARTUP=true`
- `IMAGE_PARSER_MAX_CONCURRENT_GENERATIONS=1`
- `CUDA_VISIBLE_DEVICES=0` pins the process to one visible GPU while the compose device reservation requests one NVIDIA GPU.
- Docker/Compose command includes `uvicorn ... --workers 1`

## Build

Production-like GPU image build:

```bash
docker compose build image-parser
```

Fast packaging syntax check without GPU Python extras, useful when validating Dockerfile mechanics only:

```bash
docker build --build-arg INSTALL_EXTRAS= -t image-parsing-server:base-check .
```

The base-check image is not a substitute for GPU runtime validation because it does not install the `gpu` extra.

## Run on the Linux GPU host

```bash
docker compose up -d image-parser
docker compose logs -f image-parser
```

Validate the process-level endpoints:

```bash
curl -fsS http://127.0.0.1:${IMAGE_PARSER_PORT:-8000}/healthz
curl -fsS http://127.0.0.1:${IMAGE_PARSER_PORT:-8000}/readyz
curl -fsS http://127.0.0.1:${IMAGE_PARSER_PORT:-8000}/v1/model/status
```

Only `/healthz` is a process liveness check. `/readyz` may return `503` while the resident model is loading. Treat real model/GPU success as proven only after the Linux host can start the container with NVIDIA passthrough and the service reports a loaded model or successfully handles a parser smoke request.

## GPU preflight

Run these on `ssh linux` before claiming runtime validation:

```bash
nvidia-smi
docker info --format "{{json .Runtimes}}"
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 nvidia-smi
```

If any preflight fails, report the blocker instead of treating a CPU-only build as GPU validation.
