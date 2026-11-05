from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PACKAGES = {
    "torch": ("torch", "2.12.0"),
    "torch-geometric": ("torch_geometric", "2.8.0.post1"),
    "numpy": ("numpy", "2.4.4"),
    "scipy": ("scipy", "1.17.1"),
    "networkx": ("networkx", "3.6.1"),
    "scikit-learn": ("sklearn", "1.8.0"),
    "tqdm": ("tqdm", "4.67.3"),
    "pandas": ("pandas", "3.0.2"),
    "matplotlib": ("matplotlib", "3.10.3"),
}
CODE_IMPORTS = (
    "config",
    "lp_train",
    "models.base_models",
    "models.direction_diffusion",
    "utils.data_utils",
    "hyperbolic_learning.hyperkmeans",
    "diff",
)


def main() -> int:
    errors = []
    if sys.version_info[:2] != (3, 11):
        errors.append(f"Expected Python 3.11; found {sys.version.split()[0]}")
    else:
        print(f"Python {sys.version.split()[0]}: OK")

    for distribution, (module, pinned_version) in PACKAGES.items():
        try:
            installed_version = metadata.version(distribution)
            importlib.import_module(module)
        except (metadata.PackageNotFoundError, ImportError) as exc:
            errors.append(f"{distribution}: unavailable ({exc})")
            continue
        if installed_version != pinned_version:
            errors.append(
                f"{distribution}: expected {pinned_version}, found {installed_version}"
            )
        else:
            print(f"{distribution}=={installed_version}: import OK")

    for module in CODE_IMPORTS:
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"Project import {module}: {type(exc).__name__}: {exc}")
        else:
            print(f"Project import {module}: OK")

    try:
        import torch
    except ImportError:
        pass
    else:
        print(
            "PyTorch device support: "
            f"CUDA available={torch.cuda.is_available()}, "
            f"CUDA build={torch.version.cuda}"
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
