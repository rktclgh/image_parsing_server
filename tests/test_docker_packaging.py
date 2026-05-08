from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_dockerfile_runs_single_uvicorn_worker() -> None:
    dockerfile = _read("Dockerfile")

    assert "python -m pip install -e" not in dockerfile
    assert "uvicorn" in dockerfile
    assert "--workers" in dockerfile
    assert "app.main:app" in dockerfile
    assert "IMAGE_PARSER_MODEL_LOAD_ON_STARTUP" not in dockerfile
    assert "IMAGE_PARSER_MAX_CONCURRENT_GENERATIONS=1" in dockerfile


def test_compose_requests_one_gpu_and_one_worker() -> None:
    compose = _read("compose.yaml")

    assert "capabilities: [gpu]" in compose
    assert "count: 1" in compose
    assert "- --workers\n      - \"1\"" not in compose
    assert "IMAGE_PARSER_MAX_CONCURRENT_GENERATIONS" in compose
    assert "CUDA_VISIBLE_DEVICES" in compose
    assert "healthcheck:" not in compose
    assert "IMAGE_PARSER_MODEL_LOAD_ON_STARTUP" not in compose
    gpu_device_request = compose.split("gpus:", 1)[1]
    assert "required:" not in gpu_device_request


def test_env_example_documents_resident_safe_defaults() -> None:
    env_example = _read(".env.example")

    assert "IMAGE_PARSER_VLM_MODE=resident" in env_example
    assert "IMAGE_PARSER_MODEL_LOAD_ON_STARTUP=" not in env_example
    assert "IMAGE_PARSER_MAX_CONCURRENT_GENERATIONS=1" in env_example
    assert "CUDA_VISIBLE_DEVICES=0" in env_example
    assert "TOKEN=" not in env_example


def test_dockerignore_excludes_secret_env_variants_but_keeps_example() -> None:
    dockerignore = _read(".dockerignore")

    assert ".env.*" in dockerignore
    assert "!.env.example" in dockerignore


def test_docker_service_docs_call_out_gpu_validation_boundary() -> None:
    docs = _read("docs/docker-linux-service.md")

    assert "one Uvicorn worker" in docs
    assert "/home/song/oh-my-design" not in docs
    assert "not a substitute for GPU runtime validation" in docs
    assert "docker run --rm --gpus all" in docs
    assert "IMAGE_PARSER_MODEL_LOAD_ON_STARTUP` is intentionally unset" in docs
    assert "/home/song/oh-my-design" not in _read("README.md")
