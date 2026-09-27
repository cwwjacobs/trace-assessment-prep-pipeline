"""Portable projections never carry private overlay identity or rewrite canonical bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from goldtrace_refinery.hashing import sha256_bytes
from goldtrace_refinery.portable_provenance import (
    PORTABLE_INGOT_SCHEMA,
    forbidden_projection_strings,
    matched_entry_pairs,
    project_ingot,
    project_refinery_receipt,
    serialize_portable,
)

FOUNDRY_FILE_SHA = "f" * 64


def _canonical(tmp_path: Path) -> tuple[dict, bytes, dict, bytes]:
    overlay = {
        "requested": True,
        "evaluated": True,
        "granted": True,
        "granted_status": "RIGHTS_ASSERTED",
        "ledger_path": "/home/abcd/private/rights-ledger.v2.jsonl",
        "ledger_schema": "gtdataworks.rights-ledger.v2",
        "ledger_sha256": "a" * 64,
        "bindings_used": ["event_head_sha256", "manifest_sha256"],
        "minimum_bindings": 2,
        "reason": "do not project this reason",
        "matched_entries": [
            {
                "rights_status": "OPERATOR_OWNED_FULL_RIGHTS",
                "rights_assertion_sha256": "b" * 64,
                "asserted_by": "Corey Jacobs <hello@gtdataworks.com>",
                "asserted_at": "2026-08-14T14:27:09Z",
            },
            {
                "rights_status": "MACHINE_RESULT_OPERATOR_OWNED",
                "rights_assertion_sha256": "c" * 64,
                "asserted_by": "Corey Jacobs <hello@gtdataworks.com>",
            },
        ],
        "rejected_entries": [],
        "bindings_required": {"manifest_sha256": "d" * 64, "event_head_sha256": "e" * 64},
    }
    ingot = {
        "schema_id": "goldtrace.refinery.ingot.v1",
        "ingot_id": "GTI-test",
        "source_bundle_hash": "1" * 64,
        "source_manifest_hash": "2" * 64,
        "evaluation_path": "three_tier_labyrinth_foundry_refinery",
        "refinery_status": "RIGHTS_ASSERTED",
        "mine_run_id": "run-test",
        "seal_status": "SEALED_WITH_GAPS",
        "evidence_gap_ids": ["gap-native-trace-rights-unknown"],
        "foundry_evaluation": {
            "status": "PASS",
            "receipt_hash": "3" * 64,
            "evaluation_pack_hash": "4" * 64,
            "evaluation_pack_id": "native-trace-admission",
            "evaluation_pack_version": "1.0.0",
        },
        "refinery_receipt_hash": "5" * 64,
        "rights_overlay": overlay,
    }
    receipt = {
        "stage": "refinery.cast",
        "bundle_locator_disposition": "excluded_private_local_path",
        "evaluation_path": "three_tier_labyrinth_foundry_refinery",
        "source_bundle_hash_before": "1" * 64,
        "foundry_evaluation": ingot["foundry_evaluation"],
        "verification": {
            "run_id": "run-test",
            "classification": "native_trace_import",
            "seal_status": "SEALED_WITH_GAPS",
            "entry_count": 9,
            "event_count": 21,
            "event_head_sha256": "6" * 64,
            "manifest_sha256": "2" * 64,
            "checkpoint_head": None,
            "evidence_gap_ids": ["gap-native-trace-rights-unknown"],
        },
        "transformations": ["evaluate_rights_overlay"],
        "privacy_findings_count": 0,
        "created_at": "2026-08-14T00:00:00Z",
        "rights_overlay": overlay,
        "receipt_hash": "5" * 64,
    }
    ingot_path = tmp_path / "ingot.json"
    receipt_path = tmp_path / "refinery-receipt.json"
    ingot_bytes = json.dumps(ingot, sort_keys=True).encode() + b"\n"
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode() + b"\n"
    ingot_path.write_bytes(ingot_bytes)
    receipt_path.write_bytes(receipt_bytes)
    return ingot, ingot_bytes, receipt, receipt_bytes


def test_projection_is_deterministic_and_omits_private_fields(tmp_path: Path) -> None:
    ingot, ingot_bytes, receipt, receipt_bytes = _canonical(tmp_path)
    first = project_ingot(
        ingot,
        ingot_file_sha256=sha256_bytes(ingot_bytes),
        refinery_receipt_file_sha256=sha256_bytes(receipt_bytes),
        foundry_receipt_file_sha256=FOUNDRY_FILE_SHA,
    )
    second = project_ingot(
        ingot,
        ingot_file_sha256=sha256_bytes(ingot_bytes),
        refinery_receipt_file_sha256=sha256_bytes(receipt_bytes),
        foundry_receipt_file_sha256=FOUNDRY_FILE_SHA,
    )
    assert first == second
    assert serialize_portable(first) == serialize_portable(second)
    assert first["schema_id"] == PORTABLE_INGOT_SCHEMA
    assert first["source_ingot_schema_id"] == "goldtrace.refinery.ingot.v1"
    assert first["foundry_receipt_file_sha256"] == FOUNDRY_FILE_SHA
    assert first["foundry_evaluation"]["receipt_hash"] == "3" * 64
    assert first["foundry_receipt_file_sha256"] != first["foundry_evaluation"]["receipt_hash"]
    assert first["refinery_status"] == "RIGHTS_ASSERTED"
    assert first["rights_overlay"]["rights_assertion_bindings"] == [
        {"rights_status": "MACHINE_RESULT_OPERATOR_OWNED", "assertion_sha256": "c" * 64},
        {"rights_status": "OPERATOR_OWNED_FULL_RIGHTS", "assertion_sha256": "b" * 64},
    ]
    assert forbidden_projection_strings(first) == []
    portable_receipt = project_refinery_receipt(
        receipt, receipt_file_sha256=sha256_bytes(receipt_bytes)
    )
    assert forbidden_projection_strings(portable_receipt) == []
    assert portable_receipt["canonical_receipt_hash"] == "5" * 64
    assert portable_receipt["source_refinery_receipt_file_sha256"] == sha256_bytes(receipt_bytes)


def test_canonical_bytes_are_not_rewritten(tmp_path: Path) -> None:
    ingot, ingot_bytes, receipt, receipt_bytes = _canonical(tmp_path)
    project_ingot(
        ingot,
        ingot_file_sha256=sha256_bytes(ingot_bytes),
        refinery_receipt_file_sha256=sha256_bytes(receipt_bytes),
        foundry_receipt_file_sha256=FOUNDRY_FILE_SHA,
    )
    assert (tmp_path / "ingot.json").read_bytes() == ingot_bytes
    assert (tmp_path / "refinery-receipt.json").read_bytes() == receipt_bytes


def test_matched_pairs_for_full_provenance(tmp_path: Path) -> None:
    ingot, _, _, _ = _canonical(tmp_path)
    pairs = matched_entry_pairs(ingot["rights_overlay"])
    assert ("OPERATOR_OWNED_FULL_RIGHTS", "b" * 64) in pairs
    assert ("OPERATOR_OWNED_FULL_RIGHTS", "c" * 64) not in pairs


def test_projection_sha256_covers_canonical_bytes_including_newline(tmp_path: Path) -> None:
    ingot, ingot_bytes, _, receipt_bytes = _canonical(tmp_path)
    portable = project_ingot(
        ingot,
        ingot_file_sha256=sha256_bytes(ingot_bytes),
        refinery_receipt_file_sha256=sha256_bytes(receipt_bytes),
        foundry_receipt_file_sha256=FOUNDRY_FILE_SHA,
    )
    body = {key: value for key, value in portable.items() if key != "projection_sha256"}
    encoded = (
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        + b"\n"
    )
    assert portable["projection_sha256"] == hashlib.sha256(encoded).hexdigest()
    packed = serialize_portable(portable)
    assert packed.endswith(b"\n")
    assert packed == serialize_portable(json.loads(packed))
