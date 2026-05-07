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
