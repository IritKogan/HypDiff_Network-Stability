"""Stage 2 gate: train the hyperbolic autoencoder on the train edges, report validation AUC.

Usage (from the repository root):
    python scripts/hypstab_train_autoencoder.py --config configs/hypstab/cora.toml [--run-dir DIR]

Saves in the run directory: config.toml, run_info.json, split.npz, autoencoder.pt,
autoencoder_history.json (loss + validation AUC per epoch), summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hypstab.config import load_config  # noqa: E402
from hypstab.data.graph import load_graph  # noqa: E402
from hypstab.data.split import split_edges  # noqa: E402
from hypstab.runinfo import start_run  # noqa: E402
from hypstab.training.autoencoder import train_autoencoder  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "hypstab" / "cora.toml"))
    parser.add_argument("--run-dir", default=None, help="default: <run.output_root>/<run.name>_<UTC>")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_dir, device = start_run(config, args.run_dir)
    print(f"run dir: {run_dir} | device: {device}", flush=True)

    data = config.data
    graph = load_graph(data.dataset, data.root, data.download)
    s = config.split
    split = split_edges(
        graph.edges, graph.node_count, s.train_fraction, s.val_fraction, s.pool_fraction, s.control_pairs, s.seed
    )
    np.savez_compressed(
        run_dir / "split.npz",
        train=split.train, val=split.val, pool=split.pool,
        val_negatives=split.val_negatives, control_pairs=split.control_pairs,
    )

    start = time.perf_counter()
    result = train_autoencoder(graph, split, config, device, log_every=args.log_every)
    seconds = time.perf_counter() - start

    torch.save(result.model.state_dict(), run_dir / "autoencoder.pt")
    (run_dir / "autoencoder_history.json").write_text(json.dumps(result.history), encoding="utf-8")
    summary = {
        "graph": {"name": graph.name, "nodes": graph.node_count, "edges": len(graph.edges),
                  "sha256": graph.content_sha256()},
        "split_sizes": {k: int(len(getattr(split, k)))
                        for k in ("train", "val", "pool", "val_negatives", "control_pairs")},
        "best_val_auc": result.history["best_val_auc"],
        "best_epoch": result.history["best_epoch"],
        "epochs_run": result.history["epochs_run"],
        "training_seconds": round(seconds, 1),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
