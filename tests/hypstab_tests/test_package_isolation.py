"""hypstab/ must not import upstream HypDiff code or the old stability/ package (Decision 12).

Every import in hypstab/**/*.py must be the standard library, hypstab itself, or
one of the allowed third-party libraries of the plan.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "hypstab"
ALLOWED_THIRD_PARTY = {"torch", "numpy", "scipy", "networkx", "torch_geometric", "matplotlib", "pandas"}


def _imported_top_level(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import stays inside hypstab
                continue
            names.add(node.module.split(".")[0])
    return names


def test_hypstab_imports_only_allowed_modules():
    files = sorted(PACKAGE.rglob("*.py"))
    assert files, "hypstab/ has no Python files"
    allowed = set(sys.stdlib_module_names) | ALLOWED_THIRD_PARTY | {"hypstab"}
    offenders = {
        str(path.relative_to(REPO_ROOT)): sorted(_imported_top_level(path) - allowed) for path in files
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"forbidden imports in hypstab/: {offenders}"

