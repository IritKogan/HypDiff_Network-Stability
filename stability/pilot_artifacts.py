from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stability.edge_perturbation import Edge
from stability.scoring import StabilityScore, score_recovery


def save_pilot_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as artifact:
        json.dump(payload, artifact, indent=2, sort_keys=True, allow_nan=False)


def load_pilot_artifact(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as artifact:
        payload = json.load(artifact)
    if not isinstance(payload, dict):
        raise ValueError("pilot artifact root must be a JSON object")
    return payload


def rescore_pilot_artifact(payload: dict[str, Any]) -> StabilityScore:
    removed_edges = tuple(tuple(edge) for edge in payload["removed_edges"])
    reconstructions = tuple(
        tuple(tuple(edge) for edge in reconstruction["edges"])
        for reconstruction in payload["reconstructions"]
    )
    return score_recovery(removed_edges, reconstructions)


def count_unique_edge_sets(payload: dict[str, Any]) -> int:
    edge_sets = {
        frozenset(frozenset(edge) for edge in reconstruction["edges"])
        for reconstruction in payload["reconstructions"]
    }
    return len(edge_sets)


def serialized_edges(edges: tuple[Edge, ...]) -> list[list[str]]:
    return [list(edge) for edge in edges]
