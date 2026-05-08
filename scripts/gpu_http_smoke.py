from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx


IMAGE_ENV_VAR = "IMAGE_PARSER_GPU_HTTP_SMOKE_IMAGE"
DEFAULT_EXPECTED_COLOR_TERMS = ("purple", "#8", "#9")
DEFAULT_EXPECTED_SHAPE_TERMS = ("cloud", "blob", "rounded", "icon")


def main() -> int:
    args = _parse_args()
    image_path = _resolve_image_path(args.image)
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")

    baseline_vram_mib = _nvidia_used_memory_mib()
    if baseline_vram_mib > args.max_baseline_vram_mib:
        raise SystemExit(
            f"baseline GPU memory is too high: {baseline_vram_mib} MiB "
            f"> {args.max_baseline_vram_mib} MiB"
        )

    scenarios = args.scenario
    if "all" in scenarios:
        scenarios = ["resident-warm", "cold-unload"]

    results = []
    for scenario in scenarios:
        if scenario == "resident-warm":
            results.append(_run_resident_warm(args, image_path, baseline_vram_mib))
        elif scenario == "cold-unload":
            results.append(_run_cold_unload(args, image_path, baseline_vram_mib))
        else:
            raise SystemExit(f"unsupported scenario: {scenario}")

    print(json.dumps({"baseline_vram_mib": baseline_vram_mib, "scenarios": results}, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real Linux GPU HTTP smoke checks against the FastAPI parser."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help=f"image fixture path; defaults to ${IMAGE_ENV_VAR}",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=["all", "resident-warm", "cold-unload"],
        default=None,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--quantization", choices=["8bit", "none"], default="8bit")
    parser.add_argument("--startup-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-warm-request-seconds", type=float, default=60.0)
    parser.add_argument("--max-cold-request-seconds", type=float, default=180.0)
    parser.add_argument("--max-baseline-vram-mib", type=int, default=1500)
    parser.add_argument("--max-loaded-vram-delta-mib", type=int, default=12500)
    parser.add_argument("--max-released-vram-delta-mib", type=int, default=1500)
    parser.add_argument(
        "--expected-width",
        type=int,
        default=640,
        help="expected fixture width; default matches the codex-color smoke image",
    )
    parser.add_argument(
        "--expected-height",
        type=int,
        default=640,
        help="expected fixture height; default matches the codex-color smoke image",
    )
    parser.add_argument(
        "--expected-color-term",
        action="append",
        default=None,
        help="semantic color token expected in the parser output; repeat for OR matching",
    )
    parser.add_argument(
        "--expected-shape-term",
        action="append",
        default=None,
        help="semantic shape token expected in the parser output; repeat for OR matching",
    )
    parser.add_argument("--cold-release-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--require-cold-vram-release",
        action="store_true",
        help="fail if cold unload does not return VRAM close to baseline",
    )
    args = parser.parse_args()
    args.scenario = args.scenario or ["all"]
    args.expected_color_term = args.expected_color_term or list(DEFAULT_EXPECTED_COLOR_TERMS)
    args.expected_shape_term = args.expected_shape_term or list(DEFAULT_EXPECTED_SHAPE_TERMS)
    return args


def _resolve_image_path(image_arg: Path | None) -> Path:
    image = image_arg or (Path(env_path) if (env_path := os.getenv(IMAGE_ENV_VAR)) else None)
    if image is None:
        raise SystemExit(f"provide --image or set {IMAGE_ENV_VAR}")
    return image.expanduser().resolve()


def _run_resident_warm(args: argparse.Namespace, image_path: Path, baseline_vram_mib: int) -> dict[str, Any]:
    with _ServerProcess(
        args,
        env_overrides={
            "IMAGE_PARSER_VLM_MODE": "resident",
            "IMAGE_PARSER_MODEL_LOAD_ON_STARTUP": "true",
            "IMAGE_PARSER_UNLOAD_AFTER_REQUEST": "false",
        },
    ) as server:
        server.wait_ready(timeout_seconds=args.startup_timeout_seconds)
        loaded_vram_mib = _nvidia_used_memory_mib()
        response, elapsed = _post_parse(server.base_url, image_path, args.request_timeout_seconds)
        _assert_compact_parse_response(response, args=args)
        if elapsed > args.max_warm_request_seconds:
            raise RuntimeError(f"resident warm parse took too long: {elapsed:.2f}s")
        loaded_delta_mib = loaded_vram_mib - baseline_vram_mib
        if loaded_delta_mib > args.max_loaded_vram_delta_mib:
            raise RuntimeError(f"resident loaded VRAM delta too high: {loaded_delta_mib} MiB")
        return {
            "scenario": "resident-warm",
            "request_seconds": round(elapsed, 2),
            "loaded_vram_mib": loaded_vram_mib,
            "loaded_vram_delta_mib": loaded_delta_mib,
            "summary": response["style"]["summary"],
            "warnings": response.get("warnings", []),
        }


def _run_cold_unload(args: argparse.Namespace, image_path: Path, baseline_vram_mib: int) -> dict[str, Any]:
    with _ServerProcess(
        args,
        env_overrides={
            "IMAGE_PARSER_VLM_MODE": "cold",
            "IMAGE_PARSER_MODEL_LOAD_ON_STARTUP": "false",
            "IMAGE_PARSER_UNLOAD_AFTER_REQUEST": "true",
        },
    ) as server:
        server.wait_ready(timeout_seconds=args.startup_timeout_seconds)
        response, elapsed = _post_parse(server.base_url, image_path, args.request_timeout_seconds)
        _assert_compact_parse_response(response, args=args)
        if elapsed > args.max_cold_request_seconds:
            raise RuntimeError(f"cold first request took too long: {elapsed:.2f}s")

        release_observation = _observe_vram_release(
            baseline_vram_mib,
            max_delta_mib=args.max_released_vram_delta_mib,
            timeout_seconds=args.cold_release_timeout_seconds,
        )
        if args.require_cold_vram_release and not release_observation["within_threshold"]:
            raise RuntimeError(
                "GPU memory did not release: "
                f"latest={release_observation['latest_vram_mib']} MiB "
                f"baseline={baseline_vram_mib} MiB "
                f"threshold={args.max_released_vram_delta_mib} MiB"
            )
        release_status = (
            "within_threshold"
            if release_observation["within_threshold"]
            else "retained_by_live_process"
        )
        return {
            "scenario": "cold-unload",
            "purpose": "diagnostic-only",
            "request_seconds": round(elapsed, 2),
            "released_vram_mib": release_observation["latest_vram_mib"],
            "released_vram_delta_mib": release_observation["latest_vram_mib"] - baseline_vram_mib,
            "vram_release_within_threshold": release_observation["within_threshold"],
            "vram_release_status": release_status,
            "vram_release_required": args.require_cold_vram_release,
            "vram_release_timeout_seconds": args.cold_release_timeout_seconds,
            "summary": response["style"]["summary"],
            "warnings": response.get("warnings", []),
        }


class _ServerProcess:
    def __init__(self, args: argparse.Namespace, *, env_overrides: dict[str, str]) -> None:
        port = args.port or _free_port(args.host)
        self.base_url = f"http://{args.host}:{port}"
        env = os.environ.copy()
        env.update(env_overrides)
        env.update(
            {
                "IMAGE_PARSER_MODEL_ID": args.model_id,
                "IMAGE_PARSER_QUANTIZATION": args.quantization,
                "IMAGE_PARSER_MAX_CONCURRENT_GENERATIONS": "1",
                "IMAGE_PARSER_GENERATION_MAX_NEW_TOKENS": "900",
                "IMAGE_PARSER_GENERATION_DO_SAMPLE": "false",
            }
        )
        self._log = tempfile.NamedTemporaryFile(
            "w+",
            encoding="utf-8",
            prefix="image-parser-gpu-http-smoke-",
            suffix=".log",
            delete=True,
        )
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                args.host,
                "--port",
                str(port),
                "--workers",
                "1",
            ],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )

    def __enter__(self) -> "_ServerProcess":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def wait_ready(self, *, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error = None
        with httpx.Client(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError(f"server exited early:\n{self._read_log_tail()}")
                try:
                    response = client.get(f"{self.base_url}/readyz")
                    if response.status_code == 200:
                        return
                    last_error = f"{response.status_code}: {response.text}"
                except httpx.HTTPError as exc:
                    last_error = str(exc)
                time.sleep(1.0)
        raise RuntimeError(f"server did not become ready: {last_error}\n{self._read_log_tail()}")

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        self._log.close()

    def _read_log_tail(self) -> str:
        self._log.flush()
        path = Path(self._log.name)
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-80:])


def _post_parse(base_url: str, image_path: Path, timeout_seconds: float) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    with httpx.Client(timeout=timeout_seconds) as client:
        with image_path.open("rb") as file_obj:
            response = client.post(
                f"{base_url}/v1/parse/compact",
                files={"file": (image_path.name, file_obj, "image/png")},
                headers={"x-request-id": "gpu-http-smoke"},
            )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return response.json(), elapsed


def _assert_compact_parse_response(payload: dict[str, Any], *, args: argparse.Namespace) -> None:
    semantic_text = json.dumps(payload, ensure_ascii=False).lower()
    metadata = payload.get("metadata", {})
    if metadata.get("width") != args.expected_width or metadata.get("height") != args.expected_height:
        raise RuntimeError(f"unexpected image metadata: {metadata}")
    if not payload.get("style", {}).get("summary"):
        raise RuntimeError("missing style summary")
    if payload.get("warnings"):
        raise RuntimeError(f"unexpected parser warnings: {payload['warnings']}")
    if not _contains_any_term(semantic_text, args.expected_color_term):
        raise RuntimeError(
            "response does not describe an expected color term: "
            f"{args.expected_color_term}"
        )
    if not _contains_any_term(semantic_text, args.expected_shape_term):
        raise RuntimeError(
            "response does not describe an expected shape term: "
            f"{args.expected_shape_term}"
        )
    if "raw_vlm_output" in payload or "prompt" in semantic_text:
        raise RuntimeError("response leaked raw VLM or prompt content")


def _contains_any_term(text: str, terms: list[str]) -> bool:
    normalized_text = text.lower()
    return any(term.lower() in normalized_text for term in terms)


def _observe_vram_release(
    baseline_vram_mib: int,
    *,
    max_delta_mib: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    latest = _nvidia_used_memory_mib()
    while time.monotonic() < deadline:
        latest = _nvidia_used_memory_mib()
        if latest - baseline_vram_mib <= max_delta_mib:
            return {"latest_vram_mib": latest, "within_threshold": True}
        time.sleep(1.0)
    return {"latest_vram_mib": latest, "within_threshold": False}


def _nvidia_used_memory_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            _nvidia_device_id(),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0].strip())


def _nvidia_device_id() -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return "0"
    return visible_devices.split(",")[0].strip() or "0"


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    raise SystemExit(main())
