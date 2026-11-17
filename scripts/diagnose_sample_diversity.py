"""Stage 1 diagnostic: do the K reconstructions differ? (validation protocol only)

Question (2026-11-17, conditional-diffusion proposal): p_e is almost binary at t*=7.
Is that because (a) the forward noise at t* is tiny, (b) the denoiser removes the
noise so all K samples collapse, or (c) the reverse process adds no fresh noise?

For each denoiser variant x sampler x t*, on the validation-recovery input (train
edges only; validation edges vs validation negatives; pool/control never used):
  * input_spread  : std across K of x_{t*} in the standardised tangent space
  * output_spread : std across K of the predicted x0 (same space)
  * sensitivity   : output_spread / input_spread (0 = noise fully removed)
  * displacement  : mean |mean_k x0 - x_start| (how far the denoiser moves the input)
  * decoded       : per-pair probability std across K, share of pairs with recovery
                    frequency exactly 0 / 1 / in between (tau at target FPR), hit, AUC.

Every setting is a command-line argument and is written to the output JSON together
with the resolved config, so the run is reproducible.

Example:
    python scripts/diagnose_sample_diversity.py --config configs/cora.toml \
        --out-dir artifacts/diagnostics/sample_diversity
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARIANTS = {
    # current default (Decision 8)
    "current_residual_edgerec": {},
    # original HypDiff denoiser and objective
    "upstream_generative": {"architecture": "upstream_gcn", "training": "generative",
                            "train_t_max": 200, "clip_x0": True},
}


def _setup(raw):
    import torch

    from stability.edge_split import split_edges
    from stability.experiment_config import config_from_dict
    from stability.graph_data import load_graph

    torch.set_num_threads(1)
    config = config_from_dict(raw)
    root = Path(config.data.root)
    graph = load_graph(config.data.dataset, root if root.is_absolute() else ROOT / root, config.data.download)
    s = config.split
    split = split_edges(graph.edges, graph.node_count, s.train_fraction, s.val_fraction,
                        s.pool_fraction, s.control_pairs, s.seed)
    return config, graph, split


def evaluate_grid(denoiser, encoder, context, graph, split, config, t_stars, samplers, reconstructions):
    import numpy as np
    import torch

    from stability.edge_recovery_experiment import recovery_seed
    from stability.model_training import normalized_adjacency
    from stability.sample_diversity import probability_diversity, tangent_spread
    from stability.validation_recovery import VALIDATION_SERIES_INDEX, _auc, calibrate_tau

    device = torch.device("cpu")
    rec = config.recovery
    adjacency = normalized_adjacency(graph.node_count, split.train, device)
    features = graph.features.to(device)
    pos = torch.as_tensor(split.val, device=device)
    neg = torch.as_tensor(split.val_negatives, device=device)
    labels = context.labels.to(device)
    centers = context.centers.to(device)
    rows = []
    with torch.no_grad():
        embeddings = encoder.encode(features, adjacency)[2]
        node_prototypes = centers[labels]
        tangent = denoiser.manifold.logmap(node_prototypes, embeddings, c=denoiser.c)
        x_start = denoiser.normalization(tangent)
        direction = torch.stack([denoiser.manifold.logmap0(p, c=denoiser.c) for p in centers])
        edge_index = denoiser.edge(adjacency.coalesce())
        for sampler in samplers:
            denoiser.sampler = sampler
            for t_star in t_stars:
                started = time.time()
                xts, x0s, prob_pos, prob_neg = [], [], [], []
                for k in range(reconstructions):
                    seed = recovery_seed(rec.seed, VALIDATION_SERIES_INDEX, k)
                    generator = torch.Generator(device=device).manual_seed(seed)
                    noise = denoiser.draw_noise(x_start, generator)
                    t = torch.full((x_start.shape[0],), t_star, dtype=torch.long)
                    xts.append(denoiser.q_sample(x_start, t, direction, noise, labels, None,
                                                 restrict=True).numpy())
                    generator = torch.Generator(device=device).manual_seed(seed)
                    x0 = denoiser._denoise_from_tstar(x_start, edge_index, direction, labels,
                                                      t_star, generator)
                    x0s.append(x0.numpy())
                    recovered = denoiser.manifold.expmap(denoiser.renormal(x0, tangent),
                                                         node_prototypes, c=denoiser.c)
                    if k == 0:  # sanity: identical to the production path
                        generator = torch.Generator(device=device).manual_seed(seed)
                        reference = denoiser.sample_from_tstar(embeddings, adjacency, labels, centers,
                                                               t_star, generator)
                        if not torch.allclose(reference, recovered, atol=1e-6):
                            raise RuntimeError("diagnostic path differs from sample_from_tstar")
                    prob_pos.append(encoder.decode(recovered, pos).numpy().astype(np.float64))
                    prob_neg.append(encoder.decode(recovered, neg).numpy().astype(np.float64))
                prob_pos, prob_neg = np.stack(prob_pos), np.stack(prob_neg)
                tau = calibrate_tau(prob_neg.reshape(-1), rec.target_fpr)
                xts, x0s = np.stack(xts), np.stack(x0s)
                input_spread, output_spread = tangent_spread(xts), tangent_spread(x0s)
                rows.append({
                    "sampler": sampler, "t_star": t_star,
                    "noise_coefficient": float(denoiser.minus_alphas_cumprod[t_star]),
                    "signal_coefficient": float(denoiser.sqrt_alphas_cumprod[t_star]),
                    "radial_coefficient": float(denoiser.getpri(torch.tensor([t_star]))),
                    "input_spread": input_spread,
                    "output_spread": output_spread,
                    "sensitivity": output_spread / input_spread if input_spread > 0 else None,
                    "displacement": float(np.abs(x0s.mean(axis=0) - x_start.numpy()).mean()),
                    "tau": tau,
                    "hit_rate": float((prob_pos >= tau).mean()),
                    "auc_mean_probability": _auc(prob_pos.mean(axis=0), prob_neg.mean(axis=0)),
                    **probability_diversity(prob_pos, prob_neg, tau),
                    "seconds": round(time.time() - started, 1),
                })
                print(f"  {sampler} t*={t_star}: sens {rows[-1]['sensitivity']:.3f} "
                      f"hit {rows[-1]['hit_rate']:.3f} intermediate "
                      f"{rows[-1]['positive']['share_intermediate']:.3f}", flush=True)
    return rows


def _variant(job):
    import torch

    from stability.hypdiff_models import StabilityLPModel
    from stability.model_training import build_model_args, normalized_adjacency, train_denoiser
    from stability.partial_recovery import CommunityContext

    raw, name, overrides, encoder_state, context_tensors, opts = job
    raw = json.loads(json.dumps(raw))
    raw["denoiser"].update(overrides)
    config, graph, split = _setup(raw)
    device = torch.device("cpu")
    args, _ = build_model_args(graph, config, device)
    encoder = StabilityLPModel(args)
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    context = CommunityContext(tuple(range(graph.node_count)), context_tensors["labels"],
                               context_tensors["centers"], config.model.curvature)
    train_adjacency = normalized_adjacency(graph.node_count, split.train, device)
    started = time.time()
    denoiser, history = train_denoiser(graph, split, config, device, encoder, args, train_adjacency, context)
    train_seconds = time.time() - started
    torch.save(denoiser.state_dict(), Path(opts["out_dir"]) / f"denoiser_{name}.pt")
    print(f"[{name}] trained in {train_seconds:.0f}s", flush=True)
    rows = evaluate_grid(denoiser, encoder, context, graph, split, config, opts["t_stars"],
                         opts["samplers"], opts["reconstructions"])
    return {"variant": name, "overrides": overrides, "resolved_denoiser": raw["denoiser"],
            "train_seconds": round(train_seconds, 1),
            "final_denoiser_loss": history["denoiser_loss"][-1], "rows": rows}


def main() -> None:
    import numpy as np
    import torch

    from stability.edge_recovery_experiment import environment_record
    from stability.model_training import build_context, train_encoder

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--t-stars", type=int, nargs="+", default=[3, 7, 15, 30, 50, 100, 150])
    parser.add_argument("--samplers", nargs="+", default=["one_step", "iterative"],
                        choices=["one_step", "iterative"])
    parser.add_argument("--reconstructions", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    config, graph, split = _setup(raw)
    device = torch.device("cpu")
    encoder, _, train_adjacency, history, _ = train_encoder(graph, split, config, device)
    context = build_context(graph, config, encoder, train_adjacency, device)
    torch.save({"encoder": encoder.state_dict(), "labels": context.labels, "centers": context.centers},
               args.out_dir / "encoder_context.pt")
    print(f"encoder best val AUC {history['autoencoder_best_val_auc']:.4f}", flush=True)

    opts = {"out_dir": str(args.out_dir), "t_stars": args.t_stars, "samplers": args.samplers,
            "reconstructions": args.reconstructions}
    jobs = [(raw, name, VARIANTS[name], encoder.state_dict(),
             {"labels": context.labels, "centers": context.centers}, opts) for name in args.variants]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        variants = list(pool.map(_variant, jobs))

    # no_diffusion / clamp_only are deterministic (zero spread by construction): not repeated here.
    result = {
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "resolved_config": raw,
        "environment": environment_record(device),
        "encoder_best_val_auc": history["autoencoder_best_val_auc"],
        "cluster_sizes": np.bincount(context.labels.numpy()).tolist(),
        "variants": variants,
    }
    (args.out_dir / "sample_diversity.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out_dir / 'sample_diversity.json'}")


if __name__ == "__main__":
    main()
