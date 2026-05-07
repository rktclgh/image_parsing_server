"""GPU-free parser utilities for compact design parsing."""

from app.parsers.fusion import fuse_deterministic_with_vlm
from app.parsers.vlm_compact import parse_vlm_compact_output

__all__ = ["fuse_deterministic_with_vlm", "parse_vlm_compact_output"]
