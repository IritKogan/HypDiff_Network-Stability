"""Train the models on the train edges and report the validation protocol (Stages 2-3 gates).

Usage (from the repository root):
    python scripts/hypstab_train.py --config configs/hypstab/cora.toml [--run-dir DIR]
        [--autoencoder-from RUN_DIR]   reuse a trained autoencoder (same data/split/encoder/decoder config)

Saves in the run directory: config.toml, run_info.json, split.npz, autoencoder.pt,
autoencoder_history.json (loss + validation AUC per epoch), communities.pt,
validation.json, summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import tomllib

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hypstab.config import config_to_dict, load_config  # noqa: E402
from hypstab.evaluation.validation import validation_report  # noqa: E402
from hypstab.data.graph import load_graph  # noqa: E402
from hypstab.data.split import split_edges  # noqa: E402
from hypstab.runinfo import start_run  # noqa: E402
from hypstab.models.hgcn import normalized_adjacency  # noqa: E402
from hypstab.training.autoencoder import build_autoencoder, train_autoencoder  # noqa: E402
from hypstab.training.communities import fit_communities  # noqa: E402

# A reused autoencoder must come from a run with identical values in these sections.
AUTOENCODER_SECTIONS = ("geometry", "data", "split", "encoder", "decoder")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "hypstab" / "cora.toml"))
    parser.add_argument("--run-dir", default=None, help="default: <run.output_root>/<run.name>_<UTC>")
    parser.add_argument("--autoencoder-from", default=None, help="run dir with autoencoder.pt to reuse")
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
    if args.autoencoder_from:
        source = Path(args.autoencoder_from)
        # Read the raw TOML: the source run may predate sections added later.
        previous = tomllib.loads((source / "config.toml").read_text(encoding="utf-8"))
        current = config_to_dict(config)
        differing = [name for name in AUTOENCODER_SECTIONS if previous.get(name) != current[name]]
        if differing:
            raise SystemExit(f"--autoencoder-from: config sections differ: {differing}")
        model = build_autoencoder(graph.features.shape[1], config).to(device)
        model.load_state_dict(torch.load(source / "autoencoder.pt", map_location=device))
        model.eval()
        history = json.loads((source / "autoencoder_history.json").read_text(encoding="utf-8"))
        history["reused_from"] = str(source)
        train_adjacency = normalized_adjacency(graph.node_count, split.train, device)
    else:
        result = train_autoencoder(graph, split, config, device, log_every=args.log_every)
        model, history, train_adjacency = result.model, result.history, result.train_adjacency
    seconds = time.perf_counter() - start
    torch.save(model.state_dict(), run_dir / "autoencoder.pt")
    (run_dir / "autoencoder_history.json").write_text(json.dumps(history), encoding="utf-8")

    features = graph.features.to(device)
    start = time.perf_counter()
    context = fit_communities(model, features, train_adjacency, config)
    community_seconds = time.perf_counter() - start
    torch.save({"labels": context.labels.cpu(), "centers": context.centers.cpu()}, run_dir / "communities.pt")
    validation = validation_report(model, features, train_adjacency, split, context, config)
    (run_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    summary = {
        "graph": {"name": graph.name, "nodes": graph.node_count, "edges": len(graph.edges),
                  "sha256": graph.content_sha256()},
        "split_sizes": {k: int(len(getattr(split, k)))
                        for k in ("train", "val", "pool", "val_negatives", "control_pairs")},
        "best_val_auc": history["best_val_auc"],
        "best_epoch": history["best_epoch"],
        "epochs_run": history["epochs_run"],
        "autoencoder_seconds": round(seconds, 1),
        "communities": context.summary(),
        "community_seconds": round(community_seconds, 1),
        "validation_auc": {name: value["auc"] for name, value in validation.items() if isinstance(value, dict)},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
