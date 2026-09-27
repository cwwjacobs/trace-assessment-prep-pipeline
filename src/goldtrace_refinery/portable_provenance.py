"""Deterministic portable projections of canonical Refinery artifacts.

Canonical ingot.json and refinery-receipt.json stay private and byte-identical.
These projections are the only forms eligible for a distributable pack.

This module does not re-evaluate a rights overlay. It copies already-granted
public overlay facts and the closed (rights_status, assertion_sha256) pairs
from matched_entries. A later full-provenance check may rebind these bytes to
operator-held canonical files; that is not a new rights determination.
"""

from __future__ import annotations

from typing import Any

from .hashing import canonical_json, sha256_bytes, sha256_file

PORTABLE_INGOT_SCHEMA = "gtdataworks.pack.portable-ingot.v1"
PORTABLE_RECEIPT_SCHEMA = "gtdataworks.pack.portable-refinery-receipt.v1"
PORTABLE_INGOT_RELPATH = "provenance/portable-ingot.v1.json"
PORTABLE_RECEIPT_RELPATH = "provenance/portable-refinery-receipt.v1.json"
LEGACY_INGOT_RELPATH = "provenance/ingot.json"
LEGACY_RECEIPT_RELPATH = "provenance/refinery-receipt.json"

ASSERTION_DISPOSITION = "EXTERNAL_PRIVATE_VERIFICATION_INPUT"

DISPOSITION_PRIVATE_LOCAL_PATH = "PRIVATE_LOCAL_PATH"
DISPOSITION_PRIVATE_OPERATOR_IDENTITY = "PRIVATE_OPERATOR_IDENTITY"
DISPOSITION_PRIVATE_AUDIT_ONLY = "PRIVATE_AUDIT_ONLY"

_OVERLAY_OMISSIONS = (
    {"path": "rights_overlay.ledger_path", "disposition": DISPOSITION_PRIVATE_LOCAL_PATH},
    {
        "path": "rights_overlay.matched_entries",
        "disposition": DISPOSITION_PRIVATE_OPERATOR_IDENTITY,
    },
    {
        "path": "rights_overlay.rejected_entries",
        "disposition": DISPOSITION_PRIVATE_AUDIT_ONLY,
    },
    {
        "path": "rights_overlay.bindings_required",
        "disposition": DISPOSITION_PRIVATE_AUDIT_ONLY,
    },
    {"path": "rights_overlay.reason", "disposition": DISPOSITION_PRIVATE_AUDIT_ONLY},
)


class PortableProvenanceError(ValueError):
    """A portable projection could not be built or did not recompute."""


def serialize_portable(document: dict[str, Any]) -> bytes:
    """Canonical portable bytes: sorted compact UTF-8 JSON plus one newline."""
    return canonical_json(document) + b"\n"


def file_sha256(path) -> str:
    return sha256_file(path)


def _hex64(value: Any) -> str | None:
    if isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    ):
        return value
    return None


def rights_assertion_bindings(overlay: Any) -> list[dict[str, str]]:
    """Sorted unique {rights_status, assertion_sha256} from overlay match summaries."""
    if not isinstance(overlay, dict):
        return []
    seen: set[tuple[str, str]] = set()
    for entry in overlay.get("matched_entries") or []:
        if not isinstance(entry, dict):
            continue
        status = entry.get("rights_status")
        digest = _hex64(entry.get("rights_assertion_sha256"))
        if not isinstance(status, str) or not status or digest is None:
            continue
        seen.add((status, digest))
    return [
        {"rights_status": status, "assertion_sha256": digest}
        for status, digest in sorted(seen)
    ]


def matched_entry_pairs(overlay: Any) -> set[tuple[str, str]]:
    return {
        (item["rights_status"], item["assertion_sha256"])
        for item in rights_assertion_bindings(overlay)
    }


def project_overlay(overlay: Any) -> dict[str, Any]:
    """Public overlay facts only. Never path, identity, reason, or bodies."""
    if not isinstance(overlay, dict):
        overlay = {"requested": False}
    requested = bool(overlay.get("requested"))
    bindings = rights_assertion_bindings(overlay) if requested else []
    used = overlay.get("bindings_used") if requested else []
    if not isinstance(used, list) or any(not isinstance(item, str) for item in used):
        used = []
    schema = overlay.get("ledger_schema") if requested else None
    ledger_sha = _hex64(overlay.get("ledger_sha256")) if requested else None
    granted_status = overlay.get("granted_status") if overlay.get("granted") else None
    return {
        "requested": requested,
        "evaluated": bool(overlay.get("evaluated")) if requested else False,
        "granted": bool(overlay.get("granted")) if requested else False,
        "granted_status": granted_status if isinstance(granted_status, str) else None,
        "ledger_schema": schema if isinstance(schema, str) else None,
        "ledger_sha256": ledger_sha,
        "bindings_used": list(used),
        "minimum_bindings": overlay.get("minimum_bindings") if requested else None,
        "rights_assertion_bindings": bindings,
    }


def _foundry_public(block: Any) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return None
    return {
        "status": block.get("status"),
        "receipt_hash": _hex64(block.get("receipt_hash")),
        "evaluation_pack_hash": _hex64(block.get("evaluation_pack_hash")),
        "evaluation_pack_id": block.get("evaluation_pack_id")
        if isinstance(block.get("evaluation_pack_id"), str)
        else None,
        "evaluation_pack_version": block.get("evaluation_pack_version")
        if isinstance(block.get("evaluation_pack_version"), str)
        else None,
    }


def _verification_public(block: Any) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return None
    gaps = block.get("evidence_gap_ids")
    return {
        "run_id": block.get("run_id") if isinstance(block.get("run_id"), str) else None,
        "classification": block.get("classification")
        if isinstance(block.get("classification"), str)
        else None,
        "seal_status": block.get("seal_status")
        if isinstance(block.get("seal_status"), str)
        else None,
        "entry_count": block.get("entry_count")
        if type(block.get("entry_count")) is int
        else None,
        "event_count": block.get("event_count")
        if type(block.get("event_count")) is int
        else None,
        "event_head_sha256": _hex64(block.get("event_head_sha256")),
        "manifest_sha256": _hex64(block.get("manifest_sha256")),
        "checkpoint_head": block.get("checkpoint_head"),
        "evidence_gap_ids": list(gaps) if isinstance(gaps, list) else [],
    }


def _with_projection_hash(document: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in document.items() if key != "projection_sha256"}
    digest = sha256_bytes(serialize_portable(body))
    return {**body, "projection_sha256": digest}


def project_ingot(
    ingot: dict[str, Any],
    *,
    ingot_file_sha256: str,
    refinery_receipt_file_sha256: str,
    foundry_receipt_file_sha256: str,
) -> dict[str, Any]:
    if not isinstance(ingot, dict):
        raise PortableProvenanceError("canonical ingot must be an object")
    if _hex64(foundry_receipt_file_sha256) is None:
        raise PortableProvenanceError(
            "foundry_receipt_file_sha256 must be the SHA-256 of the exact Foundry receipt file"
        )
    overlay = ingot.get("rights_overlay")
    document = {
        "schema_id": PORTABLE_INGOT_SCHEMA,
        "source_ingot_schema_id": ingot.get("schema_id")
        if isinstance(ingot.get("schema_id"), str)
        else None,
        "source_ingot_file_sha256": ingot_file_sha256,
        "source_refinery_receipt_file_sha256": refinery_receipt_file_sha256,
        "foundry_receipt_file_sha256": foundry_receipt_file_sha256,
        "ingot_id": ingot.get("ingot_id"),
        "source_bundle_hash": _hex64(ingot.get("source_bundle_hash")),
        "source_manifest_hash": _hex64(ingot.get("source_manifest_hash")),
        "evaluation_path": ingot.get("evaluation_path"),
        "refinery_status": ingot.get("refinery_status"),
        "mine_run_id": ingot.get("mine_run_id"),
        "seal_status": ingot.get("seal_status"),
        "evidence_gap_ids": list(ingot.get("evidence_gap_ids") or []),
        "foundry_evaluation": _foundry_public(ingot.get("foundry_evaluation")),
        "refinery_receipt_hash": _hex64(ingot.get("refinery_receipt_hash")),
        "rights_overlay": project_overlay(overlay),
        "omitted_fields": [dict(item) for item in _OVERLAY_OMISSIONS],
    }
    return _with_projection_hash(document)


def project_refinery_receipt(
    receipt: dict[str, Any],
    *,
    receipt_file_sha256: str,
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise PortableProvenanceError("canonical Refinery receipt must be an object")
    document = {
        "schema_id": PORTABLE_RECEIPT_SCHEMA,
        "source_refinery_receipt_file_sha256": receipt_file_sha256,
        "canonical_receipt_hash": _hex64(receipt.get("receipt_hash")),
        "stage": receipt.get("stage"),
        "bundle_locator_disposition": receipt.get("bundle_locator_disposition"),
        "evaluation_path": receipt.get("evaluation_path"),
        "source_bundle_hash_before": _hex64(receipt.get("source_bundle_hash_before")),
        "foundry_evaluation": _foundry_public(receipt.get("foundry_evaluation")),
        "verification": _verification_public(receipt.get("verification")),
        "transformations": list(receipt.get("transformations") or []),
        "privacy_findings_count": receipt.get("privacy_findings_count"),
        "created_at": receipt.get("created_at"),
        "rights_overlay": project_overlay(receipt.get("rights_overlay")),
        "omitted_fields": [dict(item) for item in _OVERLAY_OMISSIONS],
    }
    return _with_projection_hash(document)


def forbidden_projection_strings(document: dict[str, Any]) -> list[str]:
    """Return leak tokens that must never appear in a portable object."""
    rendered = serialize_portable(document).decode("utf-8")
    hits: list[str] = []
    lowered = rendered.lower()
    if "@" in rendered and "gtdataworks.com" in lowered:
        hits.append("operator_email")
    if "/home/" in rendered or "/tmp/" in rendered:
        hits.append("absolute_path")
    if '"ledger_path"' in rendered:
        hits.append("ledger_path")
    if '"matched_entries"' in rendered:
        hits.append("matched_entries")
    if '"asserted_by"' in rendered:
        hits.append("asserted_by")
    if '"asserted_at"' in rendered:
        hits.append("asserted_at")
    if '"basis"' in rendered:
        hits.append("basis")
    if '"reason":' in rendered:
        hits.append("reason")
    return hits
