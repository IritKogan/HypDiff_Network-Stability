"""Strict TOML configuration for hypstab runs.

Rules (project rules + Decision 12):
- Every key must be present in the file: there are no hidden defaults, so the
  saved file fully describes a run.
- Unknown keys are rejected (catches typos).
- Types are checked against the dataclass annotations; value ranges and choices
  are checked in each section's ``validate``.
- ``config_to_toml`` writes the resolved config; ``load_config`` of that text gives
  back an equal object (tested), so a saved run can be reproduced from its file.

New sections are added stage by stage: add a frozen dataclass, give it a
``validate`` method, and add it as a field of ``HypstabConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import math
from pathlib import Path
import tomllib
import typing
from typing import Any

DEVICES = ("auto", "cpu", "cuda")
MANIFOLDS = ("poincare",)
DATASETS = ("Cora", "CiteSeer", "PubMed")
ACTIVATIONS = ("relu", "leaky_relu", "elu", "tanh", "identity")  # hypstab/models/hgcn.py
ANGULAR_NOISE = ("upstream", "folded")                            # hypstab/diffusion/model.py
SAMPLERS = ("one_step",)                                          # "iterative" added in Stage 4b


class ConfigError(ValueError):
    """Raised for a missing, unknown, mistyped or out-of-range config value."""


@dataclass(frozen=True)
class RunSection:
    name: str
    output_root: str
    device: str
    seed: int

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("run.name must not be empty")
        _choice("run.device", self.device, DEVICES)
        if self.seed < 0:
            raise ConfigError(f"run.seed must be >= 0, got {self.seed}")


@dataclass(frozen=True)
class GeometrySection:
    manifold: str
    curvature: float

    def validate(self) -> None:
        _choice("geometry.manifold", self.manifold, MANIFOLDS)
        if not (math.isfinite(self.curvature) and self.curvature > 0):
            raise ConfigError(f"geometry.curvature must be a finite number > 0, got {self.curvature}")


@dataclass(frozen=True)
class DataSection:
    dataset: str
    root: str
    download: bool

    def validate(self) -> None:
        _choice("data.dataset", self.dataset, DATASETS)


@dataclass(frozen=True)
class SplitSection:
    seed: int
    train_fraction: float
    val_fraction: float
    pool_fraction: float
    control_pairs: int

    def validate(self) -> None:
        for name in ("train_fraction", "val_fraction", "pool_fraction"):
            value = getattr(self, name)
            if not 0 < value < 1:
                raise ConfigError(f"split.{name} must be in (0, 1), got {value}")
        total = self.train_fraction + self.val_fraction + self.pool_fraction
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ConfigError(f"split fractions must sum to 1, got {total}")
        if self.control_pairs < 1 or self.seed < 0:
            raise ConfigError("split.control_pairs must be >= 1 and split.seed >= 0")


@dataclass(frozen=True)
class EncoderSection:
    seed: int
    dim: int
    num_layers: int
    activation: str
    final_activation: str
    dropout: float
    bias: bool
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    negatives_per_positive: int

    def validate(self) -> None:
        _choice("encoder.activation", self.activation, ACTIVATIONS)
        _choice("encoder.final_activation", self.final_activation, ACTIVATIONS)
        for name in ("dim", "num_layers", "max_epochs", "patience", "negatives_per_positive"):
            if getattr(self, name) < 1:
                raise ConfigError(f"encoder.{name} must be >= 1, got {getattr(self, name)}")
        if not 0 <= self.dropout < 1:
            raise ConfigError(f"encoder.dropout must be in [0, 1), got {self.dropout}")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.seed < 0:
            raise ConfigError("encoder.learning_rate must be > 0, weight_decay >= 0, seed >= 0")


@dataclass(frozen=True)
class DecoderSection:
    fermi_dirac_r: float
    fermi_dirac_t: float

    def validate(self) -> None:
        if not (math.isfinite(self.fermi_dirac_t) and self.fermi_dirac_t > 0):
            raise ConfigError(f"decoder.fermi_dirac_t must be > 0, got {self.fermi_dirac_t}")
        if not math.isfinite(self.fermi_dirac_r):
            raise ConfigError("decoder.fermi_dirac_r must be finite")


@dataclass(frozen=True)
class CommunitySection:
    n_clusters: int
    max_iter: int
    frechet_steps: int
    frechet_tol: float
    seed: int

    def validate(self) -> None:
        for name in ("n_clusters", "max_iter", "frechet_steps"):
            if getattr(self, name) < 1:
                raise ConfigError(f"community.{name} must be >= 1, got {getattr(self, name)}")
        if not self.frechet_tol > 0 or self.seed < 0:
            raise ConfigError("community.frechet_tol must be > 0 and community.seed >= 0")


@dataclass(frozen=True)
class ClampSection:
    clip: float

    def validate(self) -> None:
        if not (math.isfinite(self.clip) and self.clip > 0):
            raise ConfigError(f"clamp.clip must be > 0, got {self.clip}")


@dataclass(frozen=True)
class DiffusionSection:
    timesteps: int
    radial_delta: float
    angular_noise: str
    clip_x0: bool

    def validate(self) -> None:
        if self.timesteps < 2:
            raise ConfigError(f"diffusion.timesteps must be >= 2, got {self.timesteps}")
        if not (math.isfinite(self.radial_delta) and self.radial_delta >= 0):
            raise ConfigError(f"diffusion.radial_delta must be >= 0, got {self.radial_delta}")
        _choice("diffusion.angular_noise", self.angular_noise, ANGULAR_NOISE)


@dataclass(frozen=True)
class DenoiserSection:
    seed: int
    hidden_dim: int
    layers: int
    epochs: int
    learning_rate: float
    train_max_drop: float
    train_t_max: int
    nonfinite_max_skips: int

    def validate(self) -> None:
        for name in ("hidden_dim", "layers", "epochs", "train_t_max"):
            if getattr(self, name) < 1:
                raise ConfigError(f"denoiser.{name} must be >= 1, got {getattr(self, name)}")
        if not 0 < self.train_max_drop < 1:
            raise ConfigError(f"denoiser.train_max_drop must be in (0, 1), got {self.train_max_drop}")
        if self.learning_rate <= 0 or self.seed < 0 or self.nonfinite_max_skips < 0:
            raise ConfigError("denoiser.learning_rate must be > 0, seed >= 0, nonfinite_max_skips >= 0")


@dataclass(frozen=True)
class RecoverySection:
    seed: int
    t_star: int
    reconstructions: int
    sampler: str

    def validate(self) -> None:
        _choice("recovery.sampler", self.sampler, SAMPLERS)
        if self.reconstructions < 1 or self.seed < 0 or self.t_star < 0:
            raise ConfigError("recovery.reconstructions must be >= 1, seed >= 0, t_star >= 0")


@dataclass(frozen=True)
class HypstabConfig:
    run: RunSection
    geometry: GeometrySection
    data: DataSection
    split: SplitSection
    encoder: EncoderSection
    decoder: DecoderSection
    community: CommunitySection
    clamp: ClampSection
    diffusion: DiffusionSection
    denoiser: DenoiserSection
    recovery: RecoverySection

    def validate(self) -> None:
        for f in fields(self):
            getattr(self, f.name).validate()
        # Cross-section checks.
        from hypstab.diffusion.schedule import DiffusionSchedule

        schedule = DiffusionSchedule(self.diffusion.timesteps, self.diffusion.radial_delta,
                                     self.geometry.curvature)
        try:
            schedule.check_t_star(self.recovery.t_star)
        except ValueError as error:
            raise ConfigError(f"recovery.t_star: {error}") from None
        if self.denoiser.train_t_max > self.diffusion.timesteps:
            raise ConfigError("denoiser.train_t_max must be <= diffusion.timesteps")
        if self.recovery.t_star >= self.denoiser.train_t_max:
            raise ConfigError(
                f"recovery.t_star = {self.recovery.t_star} must be < denoiser.train_t_max = "
                f"{self.denoiser.train_t_max} (the denoiser is trained on steps 0 .. train_t_max - 1)"
            )


# --------------------------------------------------------------------------- loading


def load_config(path: str | Path) -> HypstabConfig:
    """Read and validate a TOML config file."""
    path = Path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> HypstabConfig:
    """Build and validate a config from a (TOML-like) nested dict."""
    config = _build(HypstabConfig, raw, prefix="")
    config.validate()
    return config


def _build(cls: type, table: Any, prefix: str) -> Any:
    where = prefix.rstrip(".") or "<root>"
    if not isinstance(table, dict):
        raise ConfigError(f"{where} must be a table, got {type(table).__name__}")
    hints = typing.get_type_hints(cls)
    expected = [f.name for f in fields(cls)]
    missing = [name for name in expected if name not in table]
    unknown = sorted(set(table) - set(expected))
    if missing:
        raise ConfigError(f"missing key(s) in {where}: {', '.join(prefix + m for m in missing)}")
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {', '.join(prefix + u for u in unknown)}")
    values = {}
    for name in expected:
        kind = hints[name]
        key = prefix + name
        if is_dataclass(kind):
            values[name] = _build(kind, table[name], prefix=key + ".")
        else:
            values[name] = _check_type(key, table[name], kind)
    return cls(**values)


def _check_type(key: str, value: Any, kind: Any) -> Any:
    origin = typing.get_origin(kind)
    if origin is tuple:  # tuple[T, ...] <- TOML array
        (item_kind, _ellipsis) = typing.get_args(kind)
        if not isinstance(value, list):
            raise ConfigError(f"{key} must be an array, got {type(value).__name__}")
        return tuple(_check_type(f"{key}[{i}]", v, item_kind) for i, v in enumerate(value))
    if kind is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{key} must be true/false, got {value!r}")
        return value
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{key} must be an integer, got {value!r}")
        return value
    if kind is float:
        # An integer literal is accepted for a float key (e.g. curvature = 1).
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{key} must be a number, got {value!r}")
        return float(value)
    if kind is str:
        if not isinstance(value, str):
            raise ConfigError(f"{key} must be a string, got {value!r}")
        return value
    raise TypeError(f"unsupported config type {kind!r} for {key}")  # programming error


def _choice(key: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ConfigError(f"{key} must be one of {', '.join(map(repr, allowed))}, got {value!r}")


# --------------------------------------------------------------------------- saving


def config_to_dict(config: Any) -> dict[str, Any]:
    """Nested plain dict (tuples become lists) of a config or section."""
    out: dict[str, Any] = {}
    for f in fields(config):
        value = getattr(config, f.name)
        if is_dataclass(value):
            out[f.name] = config_to_dict(value)
        elif isinstance(value, tuple):
            out[f.name] = list(value)
        else:
            out[f.name] = value
    return out


def config_to_toml(config: HypstabConfig) -> str:
    """Resolved config as TOML text (one table per section)."""
    lines: list[str] = []
    for section, table in config_to_dict(config).items():
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        for key, value in table.items():
            if isinstance(value, dict):
                raise TypeError("nested sub-tables are not supported in config_to_toml")
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float {value!r} cannot be written to TOML")
        return repr(value)  # shortest round-trip representation
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"unsupported TOML value {value!r}")
