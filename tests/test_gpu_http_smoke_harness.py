import json
import os
import subprocess
import sys

import pytest


pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    os.getenv("IMAGE_PARSER_RUN_GPU_HTTP_SMOKE") != "1",
    reason="set IMAGE_PARSER_RUN_GPU_HTTP_SMOKE=1 to run real HTTP GPU smoke tests",
)
def test_gpu_http_smoke_harness_runs_resident_path_and_records_cold_unload_diagnostic():
    image_path = os.getenv("IMAGE_PARSER_GPU_HTTP_SMOKE_IMAGE")
    assert image_path, "set IMAGE_PARSER_GPU_HTTP_SMOKE_IMAGE to a real Linux image fixture"

    result = subprocess.run(
        [sys.executable, "scripts/gpu_http_smoke.py"],
        check=True,
        capture_output=True,
        text=True,
        timeout=420,
    )

    payload = json.loads(result.stdout)
    scenarios = {entry["scenario"]: entry for entry in payload["scenarios"]}
    assert set(scenarios) == {"resident-warm", "cold-unload"}
    assert scenarios["resident-warm"]["request_seconds"] < 60
    assert scenarios["resident-warm"]["loaded_vram_delta_mib"] <= 12500
    assert scenarios["cold-unload"]["request_seconds"] < 180
    assert scenarios["cold-unload"]["purpose"] == "diagnostic-only"
    assert scenarios["cold-unload"]["vram_release_required"] is False
    assert scenarios["cold-unload"]["vram_release_status"] in {
        "within_threshold",
        "retained_by_live_process",
    }
    assert isinstance(scenarios["cold-unload"]["vram_release_within_threshold"], bool)
    assert scenarios["cold-unload"]["released_vram_delta_mib"] >= 0
    for scenario in scenarios.values():
        assert scenario["summary"]
        assert scenario["warnings"] == []
