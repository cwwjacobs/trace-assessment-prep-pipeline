"""Production semantic review is fail-closed until delegation is authenticated."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from goldtrace_hallmark.verify import production_semantic_reviewer, verify_product_lot
from goldtrace_vault.store import LocalVault, VaultError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_vault_store import _canonical_lot_hash, _make_product_lot, _write_json


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_production_default_reviewer_holds_every_source() -> None:
    result = production_semantic_reviewer(
        {"ingot_id": "source-ingot"}, {"rubric_id": "review-v1"}
    )

    assert result["disposition"] == "HOLD"
    assert result["blocked_reason"] == "production_semantic_reviewer_not_configured"


def test_default_production_hallmark_cannot_pass_or_enter_vault(tmp_path: Path) -> None:
    lot, _product_hash = _make_product_lot(
        tmp_path,
        fixture_only=False,
        rights_status="declared",
    )
    # The fixture builder normally leaves no source ingot for semantic review.
    # One exact source is enough to exercise the production default chain; any
    # unrelated lot defects can only keep the final disposition fail-closed.
    _write_jsonl(
        lot / "provenance" / "source-ingots.jsonl",
        [{"ingot_id": "source-ingot"}],
    )
    manifest_path = lot / "product-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["product_hash"] = _canonical_lot_hash(lot)
    _write_json(manifest_path, manifest)

    report = verify_product_lot(lot, mode="production")
    semantic_check = next(
        check for check in report["checks"] if check["name"] == "semantic_rereview"
    )

    assert semantic_check["ok"] is False
    assert report["status"] != "PASS"
    assert report["semantic_reviews"] == [
        {
            "disposition": "HOLD",
            "reviewer_type": "unconfigured",
            "blocked_reason": "production_semantic_reviewer_not_configured",
            "notes": "No production semantic reviewer has been delegated by the operator.",
            "rubric_id": None,
            "source_ingot_id": "source-ingot",
        }
    ]

    vault = LocalVault(tmp_path / "vault", production_mode=True)
    with pytest.raises(VaultError, match="non-PASS"):
        vault.admit(lot, report)


def test_design_does_not_make_default_reviewer_accept() -> None:
    source = Path(production_semantic_reviewer.__code__.co_filename).read_text(
        encoding="utf-8"
    )
    start = source.index("def production_semantic_reviewer")
    end = source.index("\ndef verify_product_lot", start)

    assert '"disposition": "HOLD"' in source[start:end]
    assert '"disposition": "ACCEPT"' not in source[start:end]
