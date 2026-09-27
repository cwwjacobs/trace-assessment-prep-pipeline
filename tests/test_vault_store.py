"""Offline contract and boundary tests for goldtrace_vault.store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from goldtrace_vault.store import (
    _REQUIRED_HALLMARK_PASS_CHECKS,
    LocalVault,
    VaultError,
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_lot_hash(lot: Path) -> str:
    """Independent test implementation of the cross-stage lot hash."""

    lines: list[str] = []
    for path in sorted(
        (path for path in lot.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(lot).as_posix(),
    ):
        relative = path.relative_to(lot).as_posix()
        if relative == "product-manifest.json":
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest.pop("product_hash", None)
            digest = hashlib.sha256(_canonical_json(manifest)).hexdigest()
        else:
            digest = _sha256_file(path)
        lines.append(f"{digest}  {relative}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_product_lot(
    tmp_path: Path,
    *,
    directory_name: str = "lot",
    product_id: str = "test-product-001",
    product_version: str = "1.0.0",
    fixture_only: bool = True,
    rights_status: str = "fixture",
    payload: str = "training-content\n",
) -> tuple[Path, str]:
    lot = tmp_path / "lots" / directory_name
    lot.mkdir(parents=True)
    (lot / "dataset").mkdir()
    (lot / "dataset" / "train.jsonl").write_text(payload, encoding="utf-8")

    rights = {
        "rights_status": rights_status,
        "fixture_only": fixture_only,
    }
    rights_path = lot / "provenance" / "rights.json"
    _write_json(rights_path, rights)

    manifest = {
        "schema_id": "goldtrace.mint.product_lot.v1",
        "product_id": product_id,
        "product_version": product_version,
        "dataset": {},
        "eval": {},
        "product_spec_hash": "1" * 64,
        "curation_ledger_hash": "2" * 64,
        "source_ingot_ledger_hash": "3" * 64,
        "lineage_manifest_hash": "4" * 64,
        "rights_manifest_hash": _sha256_file(rights_path),
        "product_hash": "0" * 64,
        "fixture_only": fixture_only,
    }
    manifest_path = lot / "product-manifest.json"
    _write_json(manifest_path, manifest)
    product_hash = _canonical_lot_hash(lot)
    manifest["product_hash"] = product_hash
    _write_json(manifest_path, manifest)
    assert _canonical_lot_hash(lot) == product_hash
    return lot, product_hash


def _make_report(
    *,
    product_hash: str,
    product_id: str = "test-product-001",
    product_version: str = "1.0.0",
    status: str = "PASS",
    fixture_only: bool = True,
    reviewer_mode: str = "fixture",
    created_at: str = "2026-08-01T00:00:00Z",
) -> dict[str, Any]:
    # Derived from the production set rather than restated. A duplicated
    # copy silently stops matching the moment a check is added, which is
    # how the substance checks broke thirteen of these at once.
    required_pass_checks = tuple(sorted(_REQUIRED_HALLMARK_PASS_CHECKS))
    report: dict[str, Any] = {
        "schema_id": "goldtrace.hallmark.report.v1",
        "product_id": product_id,
        "product_version": product_version,
        "product_hash": product_hash,
        "status": status,
        "checks": [
            {"name": name, "ok": True, "detail": "test Hallmark witness"}
            for name in required_pass_checks
        ],
        "discrepancies": [],
        "semantic_reviews": [
            {
                "disposition": "ACCEPT",
                "source_ingot_id": "GTI-test-source",
                "reviewer_type": reviewer_mode,
            }
        ],
        "reviewer": {
            "type": reviewer_mode,
            "mode": reviewer_mode,
            "identity": f"{reviewer_mode}-reviewer",
        },
        "created_at": created_at,
        "fixture_only": fixture_only,
    }
    report["report_hash"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def _rehash_report(report: dict[str, Any]) -> None:
    report.pop("report_hash", None)
    report["report_hash"] = hashlib.sha256(_canonical_json(report)).hexdigest()


def _write_report(tmp_path: Path, report: dict[str, Any], name: str = "report") -> Path:
    path = tmp_path / "reports" / f"{name}.json"
    _write_json(path, report)
    return path


@pytest.mark.unit
def test_local_vault_init_creates_internal_files(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    vault = LocalVault(vault_root, production_mode=False)

    assert vault_root.is_dir()
    assert vault.objects.is_dir()
    assert vault.ledger.is_file()
    assert vault.lock_file.is_file()
    assert vault.inspect("missing") == []


@pytest.mark.fixture_e2e
def test_admit_valid_bound_fixture_report_and_inspect(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report = _make_report(product_hash=product_hash)
    report_path = _write_report(tmp_path, report)
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    entry = vault.admit(lot, report_path)

    assert entry["action"] == "admit"
    assert entry["product_id"] == "test-product-001"
    assert entry["product_version"] == "1.0.0"
    assert entry["product_lot_hash"] == product_hash
    assert entry["hallmark_report_hash"] == report["report_hash"]
    assert entry["rights_status"] == "fixture"
    assert entry["sale_eligibility"] == "fixture_only"
    storage = Path(entry["storage_location"])
    assert storage.parent == vault.objects
    assert json.loads((storage / "HALLMARK_REPORT.json").read_text()) == report
    assert vault.inspect("test-product-001") == [entry]


@pytest.mark.unit
def test_nonfixture_admission_never_grants_commercial_eligibility(
    tmp_path: Path,
) -> None:
    lot, product_hash = _make_product_lot(
        tmp_path,
        fixture_only=False,
        rights_status="declared",
    )
    report = _make_report(
        product_hash=product_hash,
        fixture_only=False,
        reviewer_mode="production",
    )
    vault = LocalVault(tmp_path / "vault", production_mode=True)

    entry = vault.admit(lot, report)

    assert entry["rights_status"] == "declared"
    assert entry["sale_eligibility"] == "held"
    assert entry["fixture_only"] is False


@pytest.mark.unit
def test_unknown_rights_do_not_block_local_admission(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(
        tmp_path,
        fixture_only=False,
        rights_status="unknown",
    )
    report = _make_report(
        product_hash=product_hash,
        fixture_only=False,
        reviewer_mode="production",
    )
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    entry = vault.admit(lot, report)

    assert entry["rights_status"] == "unknown"
    assert entry["sale_eligibility"] == "held"


@pytest.mark.unit
@pytest.mark.parametrize(
    "missing_field",
    [
        "schema_id",
        "product_id",
        "product_version",
        "product_hash",
        "checks",
        "semantic_reviews",
        "report_hash",
    ],
)
def test_report_must_satisfy_hallmark_contract(
    tmp_path: Path,
    missing_field: str,
) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report = _make_report(product_hash=product_hash)
    report.pop(missing_field)
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="schema validation"):
        vault.admit(lot, report)


@pytest.mark.unit
def test_forged_report_with_stale_self_hash_is_rejected(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report = _make_report(product_hash=product_hash)
    report["reviewer"]["identity"] = "forged-after-hallmark"
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="report_hash integrity"):
        vault.admit(lot, report)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("product_id", "other-product", "different product_id"),
        ("product_version", "9.9.9", "different product_version"),
        ("product_hash", "f" * 64, "different product_hash"),
    ],
)
def test_rehashed_report_must_match_exact_lot_identity(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report = _make_report(product_hash=product_hash)
    report[field] = value
    _rehash_report(report)
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match=message):
        vault.admit(lot, report)


@pytest.mark.unit
@pytest.mark.parametrize("status", ["FAIL", "HOLD"])
def test_vault_rejects_valid_non_pass_report(tmp_path: Path, status: str) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report = _make_report(product_hash=product_hash, status=status)
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="non-PASS"):
        vault.admit(lot, report)


@pytest.mark.unit
def test_vault_rejects_rehashed_pass_with_failed_check(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report = _make_report(product_hash=product_hash)
    report["checks"][0]["ok"] = False
    report["discrepancies"] = [
        {"check": report["checks"][0]["name"], "detail": "deliberate contradiction"}
    ]
    _rehash_report(report)
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="contradicts failed check"):
        vault.admit(lot, report)


@pytest.mark.unit
def test_vault_rejects_rehashed_pass_missing_any_hallmark_gate(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report = _make_report(product_hash=product_hash)
    report["checks"] = [
        check for check in report["checks"] if check["name"] != "no_split_leakage"
    ]
    _rehash_report(report)
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="lacks required trust-floor checks"):
        vault.admit(lot, report)


@pytest.mark.unit
def test_vault_rejects_rehashed_pass_with_reviewer_hold(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report = _make_report(product_hash=product_hash)
    report["semantic_reviews"][0]["disposition"] = "HOLD"
    _rehash_report(report)
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="non-ACCEPT semantic disposition"):
        vault.admit(lot, report)


@pytest.mark.unit
def test_payload_changed_after_hallmark_is_rejected(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report = _make_report(product_hash=product_hash)
    (lot / "dataset" / "train.jsonl").write_text("tampered\n", encoding="utf-8")
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="exact product lot"):
        vault.admit(lot, report)


@pytest.mark.unit
def test_manifest_identity_changed_after_hallmark_is_rejected(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report = _make_report(product_hash=product_hash)
    manifest_path = lot / "product-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["product_id"] = "tampered-product"
    _write_json(manifest_path, manifest)
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="exact product lot"):
        vault.admit(lot, report)


@pytest.mark.unit
def test_rights_artifact_must_match_manifest_binding(tmp_path: Path) -> None:
    lot, _ = _make_product_lot(tmp_path)
    manifest_path = lot / "product-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rights_manifest_hash"] = "f" * 64
    _write_json(manifest_path, manifest)
    manifest["product_hash"] = _canonical_lot_hash(lot)
    _write_json(manifest_path, manifest)
    report = _make_report(product_hash=manifest["product_hash"])
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="rights_manifest_hash"):
        vault.admit(lot, report)


@pytest.mark.unit
def test_production_mode_rejects_fixture_report_and_lot(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report = _make_report(product_hash=product_hash)
    vault = LocalVault(tmp_path / "vault", production_mode=True)

    with pytest.raises(VaultError, match="fixture"):
        vault.admit(lot, report)


@pytest.mark.unit
def test_product_lot_symlink_is_rejected(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (lot / "linked.txt").symlink_to(outside)
    report = _make_report(product_hash=product_hash)
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="symlink"):
        vault.admit(lot, report)


@pytest.mark.unit
def test_report_symlink_is_rejected(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report_path = _write_report(tmp_path, _make_report(product_hash=product_hash))
    link = tmp_path / "report-link.json"
    link.symlink_to(report_path)
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="report path must not be a symlink"):
        vault.admit(lot, link)


@pytest.mark.unit
def test_report_path_inside_product_lot_is_rejected(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report_path = lot / "review.json"
    _write_json(report_path, _make_report(product_hash=product_hash))
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="outside the product lot"):
        vault.admit(lot, report_path)


@pytest.mark.unit
def test_product_lot_and_vault_roots_must_not_overlap(tmp_path: Path) -> None:
    vault = LocalVault(tmp_path / "vault", production_mode=False)
    lot, product_hash = _make_product_lot(
        vault.root,
        directory_name="inside-vault",
    )
    report = _make_report(product_hash=product_hash)

    with pytest.raises(VaultError, match="must not overlap"):
        vault.admit(lot, report)


@pytest.mark.fixture_e2e
def test_exact_duplicate_is_idempotent_and_single_ledgered(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report = _make_report(product_hash=product_hash)
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    first = vault.admit(lot, report)
    second = vault.admit(lot, report)

    assert first["action"] == "admit"
    assert second["status"] == "already_admitted"
    assert second["product_lot_hash"] == first["product_lot_hash"]
    assert second["hallmark_report_hash"] == first["hallmark_report_hash"]
    assert len(vault.inspect("test-product-001")) == 1


@pytest.mark.unit
def test_existing_object_tampering_is_not_overwritten(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    report = _make_report(product_hash=product_hash)
    vault = LocalVault(tmp_path / "vault", production_mode=False)
    entry = vault.admit(lot, report)
    storage = Path(entry["storage_location"])
    (storage / "dataset" / "train.jsonl").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(VaultError, match="overwrite refused"):
        vault.admit(lot, report)

    assert (storage / "dataset" / "train.jsonl").read_text() == "tampered\n"
    assert len(vault.inspect("test-product-001")) == 1


@pytest.mark.unit
def test_different_report_cannot_replace_stored_report(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    first_report = _make_report(product_hash=product_hash)
    second_report = _make_report(
        product_hash=product_hash,
        created_at="2026-08-01T00:00:01Z",
    )
    vault = LocalVault(tmp_path / "vault", production_mode=False)
    vault.admit(lot, first_report)

    with pytest.raises(VaultError, match="different Hallmark report"):
        vault.admit(lot, second_report)


@pytest.mark.unit
def test_same_identity_different_content_is_rejected(tmp_path: Path) -> None:
    first_lot, first_hash = _make_product_lot(
        tmp_path,
        directory_name="first",
        payload="first\n",
    )
    second_lot, second_hash = _make_product_lot(
        tmp_path,
        directory_name="second",
        payload="second\n",
    )
    vault = LocalVault(tmp_path / "vault", production_mode=False)
    vault.admit(first_lot, _make_report(product_hash=first_hash))

    with pytest.raises(VaultError, match="conflicting product identity"):
        vault.admit(second_lot, _make_report(product_hash=second_hash))


@pytest.mark.unit
def test_reserved_vault_report_cannot_be_smuggled_in_lot(tmp_path: Path) -> None:
    lot, product_hash = _make_product_lot(tmp_path)
    _write_json(lot / "HALLMARK_REPORT.json", {"planted": True})
    report = _make_report(product_hash=product_hash)
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="reserved Vault attachment"):
        vault.admit(lot, report)


@pytest.mark.unit
def test_publish_always_refused(tmp_path: Path) -> None:
    vault = LocalVault(tmp_path / "vault", production_mode=False)

    with pytest.raises(VaultError, match="publication"):
        vault.publish()
    with pytest.raises(VaultError, match="publication"):
        vault.publish("anything")
