from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def sha256_tree(root: Path, exclude: set[str] | None = None) -> str:
    exclude = exclude or set()
    root = root.resolve()
    lines: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in exclude:
            continue
        lines.append(f"{sha256_file(path)}  {rel}")
    return hashlib.sha256(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")).hexdigest()


def sha256_product_lot(root: Path) -> str:
    """Return the canonical hash of an exact Mint product lot.

    The manifest participates as canonical JSON with only ``product_hash``
    removed, avoiding a self-reference while binding all other manifest data.
    Symlinks are forbidden because Hallmark must review bytes contained in the
    lot, not mutable or out-of-tree targets.
    """
    root = root.resolve(strict=True)
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"product lot may not contain symlinks: {path.relative_to(root)}")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "product-manifest.json":
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("product-manifest.json must contain a JSON object")
            manifest = dict(manifest)
            manifest.pop("product_hash", None)
            digest = sha256_json(manifest)
        else:
            digest = sha256_file(path)
        lines.append(f"{digest}  {rel}")
    return hashlib.sha256(
        ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    ).hexdigest()
