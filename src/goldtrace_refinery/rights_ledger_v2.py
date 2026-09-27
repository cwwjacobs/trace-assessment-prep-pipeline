"""Issue a v2 rights ledger by migrating frozen v1 entries onto live bindings.

v1 bytes are never rewritten. Bindings are recomputed by Labyrinth
``verify_bundle``. Commit 01bc244 is not an input. Predecessor pointers are
issuance/receipt metadata; the overlay evaluator does not consult them.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import sha256_bytes, sha256_file
from .pack_record import (
    ADMISSIBLE_RIGHTS_STATUS,
    rights_assertion_sha256,
    validate_rights_assertion,
)
from .rights_overlay import (
    LEDGER_SCHEMA,
    LEDGER_SCHEMA_V2,
    _ENTRY,
    _HEADER,
    _load_assertion_bodies,
    _load_ledger_lines,
)

#: SHA-256 of the frozen v1 ledger on origin/main (fd804a3 / e1e40c9).
V1_LEDGER_SHA256 = "7307ec19447cc442d10a5b4123353ffdfccd85b7c52448870d6e36bd49573bc6"

ISSUANCE_RECEIPT_SCHEMA = "gtdataworks.rights-ledger.v2.issuance-receipt.v1"
NATIVE_TRACE = "native_trace_import"
_HEX64 = __import__("re").compile(r"^[a-f0-9]{64}$")


class RightsLedgerV2Error(ValueError):
    """Issuance could not produce a valid v2 ledger."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _verify_bundle(bundle: Path) -> Any:
    try:
        from goldentrace.capture.seal import verify_bundle
    except ImportError as exc:
        raise RightsLedgerV2Error(
            "Labyrinth goldentrace.capture.seal.verify_bundle is required to "
            f"recompute bindings: {exc}"
        ) from exc
    return verify_bundle(bundle)


def _require_attestation(asserted_by: str, basis: str) -> None:
    if not isinstance(asserted_by, str) or not asserted_by.strip():
        raise RightsLedgerV2Error("OWNER_INPUT_REQUIRED: asserted_by is missing")
    if not isinstance(basis, str) or not basis.strip():
        raise RightsLedgerV2Error("OWNER_INPUT_REQUIRED: basis is missing")


def _fingerprint_runs(runs_root: Path, run_ids: set[str]) -> tuple[str, int]:
    lines: list[str] = []
    count = 0
    for run_id in sorted(run_ids):
        root = runs_root / run_id
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rel = path.relative_to(runs_root).as_posix()
                lines.append(f"{sha256_file(path)}  {rel}")
                count += 1
    digest = sha256_bytes("\n".join(lines).encode("utf-8") + (b"\n" if lines else b""))
    return digest, count


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(tmp, flags, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def issue_rights_ledger_v2(
    *,
    v1_path: Path | str,
    runs_root: Path | str,
    out_path: Path | str,
    receipt_path: Path | str,
    asserted_by: str,
    basis: str,
    asserted_at: str | None = None,
    assertions_path: Path | str | None = None,
) -> dict[str, Any]:
    """Migrate every v1 entry onto live verify_bundle bindings. Never writes v1."""

    _require_attestation(asserted_by, basis)
    v1_path = Path(v1_path).resolve()
    runs_root = Path(runs_root).resolve()
    out_path = Path(out_path).resolve()
    receipt_path = Path(receipt_path).resolve()

    if not v1_path.is_file():
        raise RightsLedgerV2Error(f"v1 ledger not found: {v1_path}")
    if v1_path.name != "rights-ledger.v1.jsonl":
        raise RightsLedgerV2Error(
            f"v1 path must be the frozen rights-ledger.v1.jsonl; got {v1_path.name}"
        )
    if out_path == v1_path or out_path.name == "rights-ledger.v1.jsonl":
        raise RightsLedgerV2Error("refusing to write v2 onto the v1 ledger path")
    if out_path.exists():
        raise RightsLedgerV2Error(f"v2 output already exists; refusing to replace: {out_path}")

    v1_sha = sha256_file(v1_path)
    if v1_sha != V1_LEDGER_SHA256:
        raise RightsLedgerV2Error(
            f"v1 ledger hash {v1_sha} is not the frozen pin {V1_LEDGER_SHA256}"
        )

    records = _load_ledger_lines(v1_path)
    header = next((r for r in records if r.get("record_type") == _HEADER), None)
    entries = [r for r in records if r.get("record_type") == _ENTRY]
    if header is None or (header.get("ledger_schema") or "") != LEDGER_SCHEMA:
        raise RightsLedgerV2Error("v1 header is missing or not a v1 ledger")
    if len(entries) != 75:
        raise RightsLedgerV2Error(f"v1 ledger must contain 75 entries; found {len(entries)}")

    assertion_dir = Path(assertions_path).resolve() if assertions_path else v1_path.parent
    bodies = _load_assertion_bodies(assertion_dir / v1_path.name, entries)
    assertions_file = assertion_dir / "rights-assertions.v1.json"
    if not assertions_file.is_file():
        raise RightsLedgerV2Error(f"frozen assertion sidecar not found: {assertions_file}")

    issued_at = asserted_at or _utc_now()
    attestation_body = {
        "asserted_by": asserted_by,
        "asserted_at": issued_at,
        "basis": basis,
    }
    attestation = {
        **attestation_body,
        "assertion_sha256": rights_assertion_sha256(attestation_body),
    }

    v2_entries: list[dict[str, Any]] = []
    run_reports: list[dict[str, Any]] = []
    seen_predecessor: set[tuple[str, str]] = set()
    verify_calls = 0
    cache: dict[str, Any] = {}

    for entry in entries:
        run_id = entry.get("sealed_artifact_id")
        status = entry.get("rights_status")
        assertion_digest = entry.get("rights_assertion_sha256")
        if not isinstance(run_id, str) or not run_id:
            raise RightsLedgerV2Error("v1 entry missing sealed_artifact_id")
        if status not in ADMISSIBLE_RIGHTS_STATUS:
            raise RightsLedgerV2Error(
                f"v1 entry {run_id} has inadmissible rights_status {status!r}"
            )
        predecessor_key = (run_id, str(status))
        if predecessor_key in seen_predecessor:
            raise RightsLedgerV2Error(f"duplicate v1 predecessor {predecessor_key}")
        seen_predecessor.add(predecessor_key)

        bundle = runs_root / run_id
        if run_id not in cache:
            if not bundle.is_dir():
                raise RightsLedgerV2Error(f"sealed bundle not found: {bundle}")
            verification = _verify_bundle(bundle)
            verify_calls += 1
            cache[run_id] = verification
        verification = cache[run_id]
        if verification.classification != NATIVE_TRACE:
            raise RightsLedgerV2Error(
                f"{run_id} classification is {verification.classification!r}; "
                "first v2 issuance is native_trace_import only"
            )
        v1_seal = entry.get("canonical_seal") or {}
        v1_manifest = v1_seal.get("manifest_sha256")
        if verification.manifest_sha256 != v1_manifest:
            raise RightsLedgerV2Error(
                f"{run_id} live manifest {verification.manifest_sha256} does not "
                f"match v1 {v1_manifest}"
            )
        event_head = verification.event_head_sha256
        if not isinstance(event_head, str) or _HEX64.fullmatch(event_head) is None:
            raise RightsLedgerV2Error(f"{run_id} live event_head_sha256 is not a SHA-256")

        assertion = bodies.get(assertion_digest) if isinstance(assertion_digest, str) else None
        if assertion is None:
            raise RightsLedgerV2Error(
                f"{run_id} {status} has no matching assertion body for {assertion_digest}"
            )
        errors = validate_rights_assertion(assertion)
        if errors:
            raise RightsLedgerV2Error(f"{run_id} {status} assertion invalid: {errors}")
        if rights_assertion_sha256(assertion) != assertion_digest:
            raise RightsLedgerV2Error(
                f"{run_id} {status} assertion hash does not recompute to the v1 digest"
            )
        if (
            entry.get("asserted_by") != assertion.get("asserted_by")
            or entry.get("asserted_at") != assertion.get("asserted_at")
        ):
            raise RightsLedgerV2Error(
                f"{run_id} {status} copied assertion metadata does not match the body"
            )

        v2_entries.append(
            {
                "record_type": _ENTRY,
                "ledger_schema": LEDGER_SCHEMA_V2,
                "sealed_artifact_id": run_id,
                "sealed_artifact_dir": entry.get("sealed_artifact_dir") or run_id,
                "canonical_seal": {
                    "manifest_sha256": verification.manifest_sha256,
                    "event_head_sha256": event_head,
                    "checkpoint_head": verification.checkpoint_head,
                    "run_schema_version": v1_seal.get("run_schema_version"),
                    "seal_status": verification.seal_status,
                },
                "rights_status": status,
                "rights_assertion_sha256": assertion_digest,
                "asserted_by": entry.get("asserted_by"),
                "asserted_at": entry.get("asserted_at"),
                "assertion_body_ref": entry.get("assertion_body_ref"),
                "predecessor": {
                    "predecessor_ledger_sha256": V1_LEDGER_SHA256,
                    "sealed_artifact_id": run_id,
                    "rights_status": status,
                    "rights_assertion_sha256": assertion_digest,
                },
            }
        )
        run_reports.append(
            {
                "run_id": run_id,
                "rights_status": status,
                "verify_ok": True,
                "classification": verification.classification,
                "live_manifest_sha256": verification.manifest_sha256,
                "live_event_head_sha256": event_head,
                "live_checkpoint_head": verification.checkpoint_head,
                "v1_manifest_sha256": v1_manifest,
                "predecessor_match": True,
                "status": "migrated",
                "reason": "live verify_bundle bindings written",
            }
        )

    if len(v2_entries) != 75:
        raise RightsLedgerV2Error(f"expected 75 migrated entries; produced {len(v2_entries)}")
    if verify_calls != 25:
        raise RightsLedgerV2Error(
            f"expected verify_bundle once per distinct run (25); called {verify_calls}"
        )

    v2_header = {
        "record_type": _HEADER,
        "ledger_schema": LEDGER_SCHEMA_V2,
        "ledger_version": "2.0.0",
        "predecessor_ledger_schema": LEDGER_SCHEMA,
        "predecessor_ledger_sha256": V1_LEDGER_SHA256,
        "entry_count": 75,
        "scope": NATIVE_TRACE,
        "binding_fields": ["manifest_sha256", "checkpoint_head", "event_head_sha256"],
        "overlay_semantics": (
            "This ledger is an external overlay. It does not alter, reopen, reseal, "
            "or replace any sealed artifact it references."
        ),
        "migration_attestation": attestation,
        "append_only": True,
    }
    payload = (
        "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in [v2_header, *v2_entries])
        + "\n"
    )
    _atomic_write(out_path, payload)

    v1_after = sha256_file(v1_path)
    if v1_after != V1_LEDGER_SHA256:
        raise RightsLedgerV2Error("v1 ledger hash changed during issuance")

    fingerprint, file_count = _fingerprint_runs(runs_root, {item[0] for item in seen_predecessor})
    receipt = {
        "schema_id": ISSUANCE_RECEIPT_SCHEMA,
        "predecessor_ledger_path": str(v1_path),
        "predecessor_ledger_sha256": V1_LEDGER_SHA256,
        "assertions_file_sha256": sha256_file(assertions_file),
        "migration_attestation": attestation,
        "scope": NATIVE_TRACE,
        "entry_count": 75,
        "v1_entry_count": 75,
        "verify_bundle_calls": verify_calls,
        "output_path": str(out_path),
        "output_sha256": sha256_file(out_path),
        "sealed_set_digest": fingerprint,
        "sealed_set_file_count": file_count,
        "v1_sha256_after": v1_after,
        "inputs_excluded": ["01bc244"],
        "bindings_expected_at_grant": ["event_head_sha256", "manifest_sha256"],
        "created_at": issued_at,
        "runs": run_reports,
    }
    _atomic_write(
        receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return receipt
