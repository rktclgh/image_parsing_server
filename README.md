# image_parsing_server

Internal FastAPI service for parsing design images with deterministic image analysis and a resident Gemma VLM runtime.

## Development

Use `develop` as the clean integration branch. Feature work should happen in a dedicated worktree and feature branch.

```bash
git worktree add -b feature/example ../image_parsing_server-example develop
```

Python package setup and test commands are introduced with the parser service contracts PR.

GPU/model validation belongs on the Linux server.
