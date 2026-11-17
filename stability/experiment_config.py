"""Experiment configuration loaded from a TOML file.

Every key must be present in the file: there are no hidden defaults, so the
saved config fully describes a run. Unknown keys are rejected to catch typos.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from pathlib import Path
import tomllib
from typing import Any

ATTACKS = ("random", "targeted_weighted", "targeted_deterministic")
DEVICES = ("auto", "cpu", "cuda")
NOISE_FORMULAS = ("exact", "upstream")
FINAL_ACTIVATIONS = ("relu", "leaky_relu")
DENOISER_ARCHITECTURES = ("upstream_gcn", "residual_gcn", "conditional_gcn", "conditional_residual_gcn")
CONDITIONAL_ARCHITECTURES = ("conditional_gcn", "conditional_residual_gcn")
DENOISER_TRAINING = ("generative", "damage_aware", "edge_reconstruction", "conditional")
T_SAMPLING = ("per_graph", "per_node")
ANGULAR_NOISE = ("upstream", "folded")
PARAMETERIZATIONS = ("direct", "preconditioned", "geometric")
SAMPLERS = ("iterative", "one_step", "stochastic")
EDGE_DECISIONS = ("threshold", "bernoulli", "pairwise")
TAU_MODES = ("fixed", "val_fpr")
DIFFUSION_STEPS = 200  # upstream HypDiff num_timesteps (hard-coded upstream)


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_root: str
    device: str


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    root: str
    download: bool


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    train_fraction: float
    val_fraction: float
    pool_fraction: float
    control_pairs: int


@dataclass(frozen=True)
class ModelConfig:
    seed: int
    dim: int
    num_layers: int
    curvature: float
    final_activation: str
    activation: str
    dropout: float
    bias: bool
    use_attention: bool
    fermi_dirac_r: float
    fermi_dirac_t: float


@dataclass(frozen=True)
class AutoencoderConfig:
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    negatives_per_positive: int


@dataclass(frozen=True)
class DenoiserConfig:
    epochs: int
    learning_rate: float
    noise_formula: str
    architecture: str
    hidden_dim: int
    layers: int
    training: str
    train_max_drop: float
    train_t_max: int
    clip_x0: bool
    radial_delta: float
    edge_loss_weight: float
    t_sampling: str
    angular_noise: str
    parameterization: str
    prior_variance: float


@dataclass(frozen=True)
class CommunityConfig:
    n_clusters: int
    max_epochs: int
    init: str
    seed: int


@dataclass(frozen=True)
class PerturbationConfig:
    seed: int
    damage_fractions: tuple[float, ...]
    series: int
    attacks: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryConfig:
    seed: int
    t_star: int
    tau_mode: str
    tau: float
    target_fpr: float
    sampler: str
    eta: float
    steps: int
    reconstructions: int
    edge_decision: str
    calibration_folds: int


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    autoencoder: AutoencoderConfig
    denoiser: DenoiserConfig
    community: CommunityConfig
    perturbation: PerturbationConfig
    recovery: RecoveryConfig

    def __post_init__(self) -> None:
        _validate(self)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["perturbation"]["damage_fractions"] = list(self.perturbation.damage_fractions)
        data["perturbation"]["attacks"] = list(self.perturbation.attacks)
        return data

    def with_overrides(self, **sections: dict[str, Any]) -> "ExperimentConfig":
        """Return a copy with selected keys replaced, e.g. run={"device": "cpu"}."""
        updated = {}
        for section, values in sections.items():
            current = getattr(self, section)
            known = {field.name for field in fields(current)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(f"unknown keys in [{section}]: {sorted(unknown)}")
            updated[section] = replace(current, **values)
        return replace(self, **updated)


SECTIONS = {
    "run": RunConfig,
    "data": DataConfig,
    "split": SplitConfig,
    "model": ModelConfig,
    "autoencoder": AutoencoderConfig,
    "denoiser": DenoiserConfig,
    "community": CommunityConfig,
    "perturbation": PerturbationConfig,
    "recovery": RecoveryConfig,
}


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    missing_sections = set(SECTIONS) - set(raw)
    unknown_sections = set(raw) - set(SECTIONS)
    if missing_sections or unknown_sections:
        raise ValueError(
            f"config sections: missing {sorted(missing_sections)}, "
            f"unknown {sorted(unknown_sections)}"
        )
    sections = {}
    for name, cls in SECTIONS.items():
        values = dict(raw[name])
        expected = {field.name for field in fields(cls)}
        missing = expected - set(values)
        unknown = set(values) - expected
        if missing or unknown:
            raise ValueError(
                f"[{name}]: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}"
            )
        for field in fields(cls):
            values[field.name] = _coerce(f"{name}.{field.name}", field.type, values[field.name])
        sections[name] = cls(**values)
    return ExperimentConfig(**sections)


def _coerce(key: str, type_name: str, value: Any) -> Any:
    def scalar(kind: str, item: Any) -> Any:
        if kind == "float" and isinstance(item, (int, float)) and not isinstance(item, bool):
            return float(item)
        if kind == "int" and isinstance(item, int) and not isinstance(item, bool):
            return item
        if kind == "bool" and isinstance(item, bool):
            return item
        if kind == "str" and isinstance(item, str):
            return item
        raise ValueError(f"{key}: expected {kind}, got {item!r}")

    if type_name.startswith("tuple["):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{key}: expected a list, got {value!r}")
        kind = type_name[len("tuple["):].split(",")[0].strip()
        return tuple(scalar(kind, item) for item in value)
    return scalar(type_name, value)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "rb") as handle:
        return config_from_dict(tomllib.load(handle))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate(config: ExperimentConfig) -> None:
    run, split, model = config.run, config.split, config.model
    _require(run.device in DEVICES, f"run.device must be one of {DEVICES}")
    _require(bool(run.name), "run.name must not be empty")

    fractions = (split.train_fraction, split.val_fraction, split.pool_fraction)
    _require(all(0.0 < value < 1.0 for value in fractions), "split fractions must be in (0, 1)")
    _require(math.isclose(sum(fractions), 1.0), "split fractions must sum to 1")
    _require(split.control_pairs > 0, "split.control_pairs must be positive")

    _require(model.dim > 0 and model.num_layers >= 2, "model.dim > 0 and model.num_layers >= 2")
    _require(math.isclose(model.curvature, 1.0), "model.curvature must be 1.0 (pilot limitation)")
    _require(model.final_activation in FINAL_ACTIVATIONS, f"model.final_activation in {FINAL_ACTIVATIONS}")
    _require(0.0 <= model.dropout < 1.0, "model.dropout must be in [0, 1)")
    _require(bool(model.activation), "model.activation must not be empty")
    _require(model.fermi_dirac_t > 0, "model.fermi_dirac_t must be positive")

    ae = config.autoencoder
    _require(ae.max_epochs > 0 and ae.patience > 0, "autoencoder epochs/patience must be positive")
    _require(ae.learning_rate > 0, "autoencoder.learning_rate must be positive")
    _require(ae.weight_decay >= 0, "autoencoder.weight_decay must be non-negative")
    _require(ae.negatives_per_positive > 0, "autoencoder.negatives_per_positive must be positive")

    den = config.denoiser
    _require(den.epochs > 0 and den.learning_rate > 0, "denoiser epochs/learning_rate must be positive")
    _require(den.noise_formula in NOISE_FORMULAS, f"denoiser.noise_formula in {NOISE_FORMULAS}")
    _require(den.architecture in DENOISER_ARCHITECTURES, f"denoiser.architecture in {DENOISER_ARCHITECTURES}")
    _require(den.training in DENOISER_TRAINING, f"denoiser.training in {DENOISER_TRAINING}")
    _require(den.hidden_dim > 0 and den.layers > 0, "denoiser.hidden_dim and layers must be positive")
    _require(den.radial_delta >= 0.0, "denoiser.radial_delta must be non-negative")
    _require(0.0 <= den.train_max_drop < 1.0, "denoiser.train_max_drop must be in [0, 1)")
    _require(0 < den.train_t_max <= DIFFUSION_STEPS, f"denoiser.train_t_max must be in (0, {DIFFUSION_STEPS}]")
    _require(den.edge_loss_weight >= 0.0, "denoiser.edge_loss_weight must be non-negative")
    _require(den.t_sampling in T_SAMPLING, f"denoiser.t_sampling in {T_SAMPLING}")
    _require(den.angular_noise in ANGULAR_NOISE, f"denoiser.angular_noise in {ANGULAR_NOISE}")
    _require(den.parameterization in PARAMETERIZATIONS, f"denoiser.parameterization in {PARAMETERIZATIONS}")
    _require(den.prior_variance >= 0.0, "denoiser.prior_variance must be non-negative (0 = estimate)")
    _require(
        den.parameterization == "direct"
        or (den.training == "conditional" and den.architecture == "conditional_residual_gcn"),
        "denoiser.parameterization 'preconditioned' / 'geometric' needs training 'conditional' and "
        "architecture 'conditional_residual_gcn'",
    )
    _require(
        not (den.training == "generative" and den.architecture in CONDITIONAL_ARCHITECTURES),
        "conditional architectures need a damaged-graph condition: training must not be 'generative'",
    )

    com = config.community
    _require(com.n_clusters >= 2 and com.max_epochs > 0, "community.n_clusters >= 2, max_epochs > 0")
    _require(com.init in ("kmeans++", "upstream"), "community.init must be 'kmeans++' or 'upstream'")

    pert = config.perturbation
    _require(len(pert.damage_fractions) > 0, "perturbation.damage_fractions must not be empty")
    _require(
        all(0.0 < value < 1.0 for value in pert.damage_fractions),
        "damage fractions must be in (0, 1)",
    )
    _require(
        list(pert.damage_fractions) == sorted(set(pert.damage_fractions)),
        "damage fractions must be strictly increasing (cumulative series)",
    )
    _require(
        max(pert.damage_fractions) <= split.pool_fraction,
        "max damage fraction must not exceed split.pool_fraction",
    )
    _require(pert.series > 0, "perturbation.series must be positive")
    _require(
        len(pert.attacks) > 0 and set(pert.attacks) <= set(ATTACKS)
        and len(set(pert.attacks)) == len(pert.attacks),
        f"perturbation.attacks must be unique values from {ATTACKS}",
    )

    rec = config.recovery
    _require(0 <= rec.t_star < DIFFUSION_STEPS, f"recovery.t_star must be in [0, {DIFFUSION_STEPS - 1}]")
    _require(0.0 <= rec.tau <= 1.0, "recovery.tau must be in [0, 1]")
    _require(rec.tau_mode in TAU_MODES, f"recovery.tau_mode in {TAU_MODES}")
    _require(rec.sampler in SAMPLERS, f"recovery.sampler in {SAMPLERS}")
    _require(0.0 <= rec.eta <= 1.0, "recovery.eta must be in [0, 1]")
    _require(rec.steps >= 1, "recovery.steps must be positive")
    _require(rec.edge_decision in EDGE_DECISIONS, f"recovery.edge_decision in {EDGE_DECISIONS}")
    _require(rec.calibration_folds >= 2, "recovery.calibration_folds must be >= 2")
    _require(0.0 < rec.target_fpr < 1.0, "recovery.target_fpr must be in (0, 1)")
    _require(rec.reconstructions > 0, "recovery.reconstructions must be positive")
