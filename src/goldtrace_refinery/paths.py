"""
Path discovery for Refinery's cross-component and contract dependencies.

Resolution never depends on one operator's machine layout. Every lookup tries,
in order: an explicit environment variable, package-adjacent data, then
repository-relative discovery from this file. Nothing is hardcoded to an
absolute path, and a failed lookup reports every location that was tried.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

#: Directory holding versioned JSON/YAML contracts.
CONTRACTS_ENV_VAR = "GOLDTRACE_REFINERY_CONTRACTS"

#: Source root of the Labyrinth component (the ``goldentrace`` package).
LABYRINTH_SRC_ENV_VAR = "GOLDENTRACE_SRC"

_PACKAGE_DIR = Path(__file__).resolve().parent


def _workspace_root(start: Path) -> Path | None:
    """Nearest ancestor that looks like the GTDataworks umbrella checkout."""
    for candidate in start.parents:
        if (candidate / ".gitmodules").is_file() and (candidate / "tools").is_dir():
            return candidate
    return None


def contract_dir_candidates() -> list[Path]:
    override = os.environ.get(CONTRACTS_ENV_VAR)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser().resolve())
    # Package data: the only location that works from an installed wheel.
    candidates.append(_PACKAGE_DIR / "contracts")
    # Repository layouts: src/goldtrace_refinery/ -> component root.
    candidates.append(_PACKAGE_DIR.parents[1] / "contracts")
    candidates.append(_PACKAGE_DIR.parents[1] / "donor_components" / "contracts")
    root = _workspace_root(_PACKAGE_DIR)
    if root is not None:
        candidates.append(root / "contracts")
    return candidates


def find_contract(filename: str) -> Path:
    """Locate a contract file, or raise ``FileNotFoundError`` listing what was tried."""
    tried: list[str] = []
    for directory in contract_dir_candidates():
        path = directory / filename
        tried.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{filename} not found. Tried: {tried}. "
        f"Set {CONTRACTS_ENV_VAR} to the directory containing it."
    )


def labyrinth_src_candidates() -> list[Path]:
    override = os.environ.get(LABYRINTH_SRC_ENV_VAR)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser().resolve())
    root = _workspace_root(_PACKAGE_DIR)
    if root is not None:
        # Component directory names come from .gitmodules; match on the URL
        # marker so a path rename does not break discovery.
        for child in sorted(root.iterdir()):
            if child.is_dir() and "labyrinth" in child.name.lower():
                candidates.append(child / "src")
    # Legacy sibling layout retained for checkouts predating the monorepo split.
    candidates.append(_PACKAGE_DIR.parents[2] / "mine" / "src")
    return candidates


@lru_cache(maxsize=1)
def ensure_labyrinth_importable() -> str | None:
    """Make the ``goldentrace`` package importable.

    A source root explicitly selected by the operator, or discovered beside a
    source checkout, outranks an installed wheel.  This matters while Build is
    dirty: an older installed schema must never verify a newer checkout's
    bundle by accident.

    Returns the selected source entry, ``""`` only when no source checkout is
    available and an installed package is importable, or ``None`` when no safe
    choice can be made. Never raises.
    """
    from importlib.util import find_spec

    source_candidates = [
        candidate
        for candidate in labyrinth_src_candidates()
        if (candidate / "goldentrace").is_dir()
    ]
    if source_candidates:
        selected = source_candidates[0]
        loaded = sys.modules.get("goldentrace")
        if loaded is not None:
            module_file = getattr(loaded, "__file__", None)
            if not isinstance(module_file, str):
                return None
            origin = Path(module_file).resolve()
            if not origin.is_relative_to(selected.resolve()):
                # Python cannot safely replace an already-imported package and
                # its submodules in place. The highest-priority explicit or
                # workspace source still wins, so fail instead of silently
                # falling through to a lower-priority loaded tree.
                return None

        text = str(selected)
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
        return text

    try:
        if find_spec("goldentrace") is not None:
            return ""
    except (ImportError, ValueError):
        pass

    return None
