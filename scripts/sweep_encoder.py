"""Encoder hyperparameter sweep, selected on validation AUC only.

Trains only the encoder (train edges, early stopping on val edges vs val
negatives) for every combination in the grid and writes a JSON table. Pool
edges and control pairs play no part in the selection.

Example:
    python scripts/sweep_encoder.py --config configs/cora.toml \
        --set model.dim=16,32,64 --set model.dropout=0.0,0.2 --out sweep.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _parse_value(text: str):
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _run(job):
    import torch

    from stability.edge_split import split_edges
    from stability.experiment_config import config_from_dict
    from stability.graph_data import load_graph
    from stability.model_training import train_encoder

    torch.set_num_threads(1)
    raw, overrides = job
    for key, value in overrides.items():
        section, name = key.split(".")
        raw[section][name] = value
    config = config_from_dict(raw)
    data_root = Path(config.data.root)
    graph = load_graph(config.data.dataset, data_root if data_root.is_absolute() else ROOT / data_root,
                       config.data.download)
    s = config.split
    split = split_edges(graph.edges, graph.node_count, s.train_fraction, s.val_fraction,
                        s.pool_fraction, s.control_pairs, s.seed)
    started = time.time()
    _, _, _, history, _ = train_encoder(graph, split, config, torch.device("cpu"))
    return {
        **overrides,
        "best_val_auc": history["autoencoder_best_val_auc"],
        "best_epoch": history["autoencoder_best_epoch"],
        "epochs_run": history["autoencoder_epochs_run"],
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], help="section.key=v1,v2,...")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    grid = {}
    for item in args.set:
        key, values = item.split("=", 1)
        grid[key] = [_parse_value(v) for v in values.split(",")]
    keys = list(grid)
    jobs = [(json.loads(json.dumps(raw)), dict(zip(keys, combo)))
            for combo in itertools.product(*grid.values())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_run, jobs))
    results.sort(key=lambda row: -row["best_val_auc"])
    args.out.write_text(json.dumps({"config": str(args.config), "grid": grid, "results": results}, indent=2))
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
