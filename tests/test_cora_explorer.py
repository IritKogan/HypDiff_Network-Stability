"""Tests for scripts/build_cora_explorer.py helpers (no checkpoints / GPU needed)."""

import base64
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_cora_explorer", ROOT / "scripts" / "build_cora_explorer.py")
explorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explorer)


def _decode_like_the_page(b64: str, n: int) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.array([(raw[i >> 3] >> (i & 7)) & 1 for i in range(n)], dtype=bool)


def test_bitsets_round_trip_with_the_page_decoder():
    rng = np.random.default_rng(0)
    for n in (1, 7, 8, 9, 1583):
        values = rng.random(n) < 0.6
        assert np.array_equal(_decode_like_the_page(explorer._bits(values), n), values)


def test_round_keeps_missing_values_as_none():
    assert explorer._round([0.123456, np.nan], 3) == [0.123, None]


def test_template_has_the_data_placeholder_once():
    template = (ROOT / "scripts" / "cora_explorer_template.html").read_text(encoding="utf-8")
    assert template.count("/*__DATA__*/null") == 1
    assert template.startswith("<!DOCTYPE html>")
