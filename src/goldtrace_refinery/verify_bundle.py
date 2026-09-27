from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import LABYRINTH_SRC_ENV_VAR, ensure_labyrinth_importable


@dataclass
class BundleVerifyResult:
    ok: bool
    run_id: str | None
    seal_status: str | None
    manifest_sha256: str | None
    event_count: int | None
    evidence_gap_ids: list[str]
    error: str | None = None
    details: dict[str, Any] | None = None


def verify_mine_bundle(bundle_root: Path) -> BundleVerifyResult:
    """Verify a sealed Mine/Arena bundle without modifying it."""
    root = Path(bundle_root).resolve()
    if not root.is_dir():
        return BundleVerifyResult(False, None, None, None, None, [], f"not a directory: {root}")

    required = ["run.json", "manifest.json", "manifest.sha256", "events/events.ndjson"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        return BundleVerifyResult(
            False, None, None, None, None, [], f"missing payload files: {missing}"
        )

    labyrinth_source = ensure_labyrinth_importable()
    if labyrinth_source is None:
        return BundleVerifyResult(
            False,
            None,
            None,
            None,
            None,
            [],
            "labyrinth package source is unavailable or conflicts with an already "
            "imported goldentrace package; start a clean process or set "
            f"{LABYRINTH_SRC_ENV_VAR} to the intended src directory",
        )
    try:
        from goldentrace.capture.seal import verify_bundle
    except Exception as exc:  # pragma: no cover
        return BundleVerifyResult(
            False,
            None,
            None,
            None,
            None,
            [],
            f"labyrinth package (goldentrace) unavailable: {exc}. "
            f"Install the Labyrinth component or set {LABYRINTH_SRC_ENV_VAR} to its src directory.",
        )

    try:
        result = verify_bundle(root)
    except Exception as exc:
        return BundleVerifyResult(False, None, None, None, None, [], f"{type(exc).__name__}: {exc}")

    return BundleVerifyResult(
        ok=True,
        run_id=result.run_id,
        seal_status=result.seal_status,
        manifest_sha256=result.manifest_sha256,
        event_count=result.event_count,
        evidence_gap_ids=list(result.evidence_gap_ids or ()),
        details=asdict(result) if hasattr(result, "__dataclass_fields__") else result.__dict__,
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
