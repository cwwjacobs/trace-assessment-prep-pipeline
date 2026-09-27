"""v2 issuance migrates frozen v1 entries onto live verify_bundle bindings."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from goldtrace_refinery.pack_record import rights_assertion_sha256
from goldtrace_refinery.rights_ledger_v2 import (
    V1_LEDGER_SHA256,
    RightsLedgerV2Error,
    issue_rights_ledger_v2,
)
from goldtrace_refinery.rights_overlay import (
    LEDGER_SCHEMA,
    LEDGER_SCHEMA_V2,
    evaluate_rights_overlay,
    load_rights_overlay,
)

STATUSES = (
    "OPERATOR_OWNED_FULL_RIGHTS",
    "MACHINE_RESULT_OPERATOR_OWNED",
    "TRANSFORMED_RELEASE_TEXT_ONLY_SOURCE_MATERIAL_EXCLUDED",
)
BASIS = (
    "Authorize issuance of rights-ledger.v2 by carrying forward the existing "
    "v1 rights classifications and assertion references onto independently "
    "recomputed sealed-bundle bindings."
)
BY = "Corey Jacobs <hello@gtdataworks.com>"


def _assertion(basis: str, by: str = BY) -> dict:
    body = {"asserted_by": by, "asserted_at": "2026-08-14T14:27:09Z", "basis": basis}
    return {**body, "assertion_sha256": rights_assertion_sha256(body)}


def _write_v1(tmp_path: Path, entries: list[dict], assertions: dict) -> Path:
    ledger = tmp_path / "rights-ledger.v1.jsonl"
    header = {
        "record_type": "ledger_header",
        "ledger_schema": LEDGER_SCHEMA,
        "ledger_version": "1.0.0",
        "entry_count": len(entries),
    }
    lines = [json.dumps(header)] + [json.dumps(entry) for entry in entries]
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "rights-assertions.v1.json").write_text(
        json.dumps({"assertions": assertions}), encoding="utf-8"
    )
    return ledger


def _v1_entry(run_id: str, status: str, assertion: dict, manifest: str) -> dict:
    return {
        "record_type": "rights_entry",
        "ledger_schema": LEDGER_SCHEMA,
        "sealed_artifact_id": run_id,
        "sealed_artifact_dir": run_id,
        "canonical_seal": {
            "manifest_sha256": manifest,
            "checkpoint_head": None,
            "run_schema_version": "goldentrace.run.v7",
            "seal_status": "SEALED_WITH_GAPS",
        },
        "rights_status": status,
        "rights_assertion_sha256": assertion["assertion_sha256"],
        "asserted_by": assertion["asserted_by"],
        "asserted_at": assertion["asserted_at"],
        "assertion_body_ref": "sidecars/rights-assertions.v1.json",
    }


def _full_v1(tmp_path: Path) -> tuple[Path, Path, dict[str, dict]]:
    assertions = {status: {"rights_assertion": _assertion(status)} for status in STATUSES}
    entries = []
    manifests = {}
    runs = tmp_path / "runs"
    for i in range(25):
        run_id = f"run-fake-{i:02d}"
        manifest = f"{i:064x}"
        manifests[run_id] = manifest
        (runs / run_id).mkdir(parents=True)
        (runs / run_id / "run.json").write_text("{}", encoding="utf-8")
        for status in STATUSES:
            entries.append(_v1_entry(run_id, status, assertions[status]["rights_assertion"], manifest))
    return _write_v1(tmp_path, entries, assertions), runs, manifests


def _fake_verify(manifests: dict[str, str], *, event_head: str | None = None, classification: str = "native_trace_import"):
    def _verify(bundle: Path):
        run_id = bundle.name
        head = event_head if event_head is not None else ("c" * 64)
        return SimpleNamespace(
            classification=classification,
            manifest_sha256=manifests[run_id],
            event_head_sha256=head,
            checkpoint_head=None,
            seal_status="SEALED_WITH_GAPS",
        )

    return _verify


def test_v1_committed_ledger_sha256_is_frozen() -> None:
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "state" / "cli-ore-import" / "sidecars" / "rights-ledger.v1.jsonl"
        if candidate.is_file():
            import hashlib

            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            assert digest == V1_LEDGER_SHA256
            return
    pytest.skip("frozen v1 ledger is not present in this checkout")


def test_issue_refuses_wrong_v1_hash(tmp_path: Path) -> None:
    ledger = tmp_path / "rights-ledger.v1.jsonl"
    ledger.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RightsLedgerV2Error, match="frozen pin"):
        issue_rights_ledger_v2(
            v1_path=ledger,
            runs_root=tmp_path / "runs",
            out_path=tmp_path / "rights-ledger.v2.jsonl",
            receipt_path=tmp_path / "receipt.json",
            asserted_by=BY,
            basis=BASIS,
        )


def test_issue_refuses_missing_attestation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    v1, runs, manifests = _full_v1(tmp_path)
    monkeypatch.setattr(
        "goldtrace_refinery.rights_ledger_v2.V1_LEDGER_SHA256",
        __import__("hashlib").sha256(v1.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "goldtrace_refinery.rights_ledger_v2._verify_bundle", _fake_verify(manifests)
    )
    with pytest.raises(RightsLedgerV2Error, match="OWNER_INPUT_REQUIRED"):
        issue_rights_ledger_v2(
            v1_path=v1,
            runs_root=runs,
            out_path=tmp_path / "rights-ledger.v2.jsonl",
            receipt_path=tmp_path / "receipt.json",
            asserted_by="",
            basis=BASIS,
        )


def test_issue_refuses_out_path_equal_to_v1(tmp_path: Path) -> None:
    v1 = tmp_path / "rights-ledger.v1.jsonl"
    v1.write_text("x\n", encoding="utf-8")
    with pytest.raises(RightsLedgerV2Error, match="v1 ledger path"):
        issue_rights_ledger_v2(
            v1_path=v1,
            runs_root=tmp_path,
            out_path=v1,
            receipt_path=tmp_path / "receipt.json",
            asserted_by=BY,
            basis=BASIS,
        )


def test_issue_refuses_if_out_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    v1, runs, manifests = _full_v1(tmp_path)
    monkeypatch.setattr(
        "goldtrace_refinery.rights_ledger_v2.V1_LEDGER_SHA256",
        __import__("hashlib").sha256(v1.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "goldtrace_refinery.rights_ledger_v2._verify_bundle", _fake_verify(manifests)
    )
    out = tmp_path / "rights-ledger.v2.jsonl"
    out.write_text("already\n", encoding="utf-8")
    with pytest.raises(RightsLedgerV2Error, match="already exists"):
        issue_rights_ledger_v2(
            v1_path=v1,
            runs_root=runs,
            out_path=out,
            receipt_path=tmp_path / "receipt.json",
            asserted_by=BY,
            basis=BASIS,
        )


def test_issue_recomputes_heads_via_verify_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    v1, runs, manifests = _full_v1(tmp_path)
    monkeypatch.setattr(
        "goldtrace_refinery.rights_ledger_v2.V1_LEDGER_SHA256",
        __import__("hashlib").sha256(v1.read_bytes()).hexdigest(),
    )
    calls: list[str] = []

    def _verify(bundle: Path):
        calls.append(bundle.name)
        return _fake_verify(manifests)(bundle)

    monkeypatch.setattr("goldtrace_refinery.rights_ledger_v2._verify_bundle", _verify)
    out = tmp_path / "rights-ledger.v2.jsonl"
    receipt = issue_rights_ledger_v2(
        v1_path=v1,
        runs_root=runs,
        out_path=out,
        receipt_path=tmp_path / "receipt.json",
        asserted_by=BY,
        basis=BASIS,
    )
    assert len(calls) == 25
    assert receipt["verify_bundle_calls"] == 25
    assert receipt["entry_count"] == 75
    assert receipt["inputs_excluded"] == ["01bc244"]
    assert v1.read_bytes()  # still there
    overlay = load_rights_overlay(out)
    assert overlay["schema"] == LEDGER_SCHEMA_V2
    sample = overlay["entries"][0]
    assert sample["canonical_seal"]["event_head_sha256"] == "c" * 64
    assert "predecessor" in sample
    result = evaluate_rights_overlay(
        overlay,
        manifest_sha256=sample["canonical_seal"]["manifest_sha256"],
        checkpoint_head=None,
        event_head_sha256="c" * 64,
    )
    assert result.granted is True
    assert result.bindings_used == ["event_head_sha256", "manifest_sha256"]


def test_issue_refuses_non_native_trace_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    v1, runs, manifests = _full_v1(tmp_path)
    monkeypatch.setattr(
        "goldtrace_refinery.rights_ledger_v2.V1_LEDGER_SHA256",
        __import__("hashlib").sha256(v1.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "goldtrace_refinery.rights_ledger_v2._verify_bundle",
        _fake_verify(manifests, classification="live_interactive"),
    )
    with pytest.raises(RightsLedgerV2Error, match="native_trace_import"):
        issue_rights_ledger_v2(
            v1_path=v1,
            runs_root=runs,
            out_path=tmp_path / "rights-ledger.v2.jsonl",
            receipt_path=tmp_path / "receipt.json",
            asserted_by=BY,
            basis=BASIS,
        )


def test_issue_refuses_missing_event_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    v1, runs, manifests = _full_v1(tmp_path)
    monkeypatch.setattr(
        "goldtrace_refinery.rights_ledger_v2.V1_LEDGER_SHA256",
        __import__("hashlib").sha256(v1.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "goldtrace_refinery.rights_ledger_v2._verify_bundle",
        _fake_verify(manifests, event_head=""),
    )
    with pytest.raises(RightsLedgerV2Error, match="event_head"):
        issue_rights_ledger_v2(
            v1_path=v1,
            runs_root=runs,
            out_path=tmp_path / "rights-ledger.v2.jsonl",
            receipt_path=tmp_path / "receipt.json",
            asserted_by=BY,
            basis=BASIS,
        )


def test_evaluate_ignores_predecessor_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    v1, runs, manifests = _full_v1(tmp_path)
    monkeypatch.setattr(
        "goldtrace_refinery.rights_ledger_v2.V1_LEDGER_SHA256",
        __import__("hashlib").sha256(v1.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "goldtrace_refinery.rights_ledger_v2._verify_bundle", _fake_verify(manifests)
    )
    out = tmp_path / "rights-ledger.v2.jsonl"
    issue_rights_ledger_v2(
        v1_path=v1,
        runs_root=runs,
        out_path=out,
        receipt_path=tmp_path / "receipt.json",
        asserted_by=BY,
        basis=BASIS,
    )
    lines = out.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    entries = [json.loads(line) for line in lines[1:]]
    for entry in entries:
        entry["predecessor"] = {"tampered": True}
    out.write_text(
        "\n".join(json.dumps(row) for row in [header, *entries]) + "\n",
        encoding="utf-8",
    )
    # Rewriting a test fixture is fine; production issuance refuses replace.
    overlay = load_rights_overlay(out)
    result = evaluate_rights_overlay(
        overlay,
        manifest_sha256=entries[0]["canonical_seal"]["manifest_sha256"],
        checkpoint_head=None,
        event_head_sha256="c" * 64,
    )
    assert result.granted is True
