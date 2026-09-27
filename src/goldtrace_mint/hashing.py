from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def sha256_tree(root: Path) -> str:
    root = root.resolve()
    lines: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    return sha256_bytes(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))


def sha256_product_lot(root: Path) -> str:
    """Hash an entire product lot without a self-referential manifest hash.

    Every regular file participates.  ``product-manifest.json`` contributes
    canonical JSON with only its ``product_hash`` field omitted.  Symlinks are
    rejected so hashing cannot escape the lot or change meaning later.
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
    return sha256_bytes(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
