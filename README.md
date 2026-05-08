# image_parsing_server

Internal FastAPI service for parsing design images with deterministic image analysis and a resident Gemma VLM runtime.

## Development

Use `develop` as the clean integration branch. Feature work should happen in a dedicated worktree and feature branch.

```bash
git worktree add -b feature/example ../image_parsing_server-example develop
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

GPU/model validation belongs on the Linux server.

## Docker Linux GPU service

Docker packaging is available for the Linux GPU host. The service is intentionally configured for one Uvicorn worker so Gemma is loaded once in VRAM.

```bash
ssh linux
cd /home/song/oh-my-design/image_parsing_server-docker-packaging
cp .env.example .env
docker compose build image-parser
docker compose up -d image-parser
curl -fsS http://127.0.0.1:${IMAGE_PARSER_PORT:-8000}/healthz
```

See [docs/docker-linux-service.md](docs/docker-linux-service.md) for worktree setup, environment defaults, GPU preflight checks, and runtime validation notes.
