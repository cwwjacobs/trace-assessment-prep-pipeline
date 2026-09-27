"""External rights overlay evaluation for sealed native-trace imports.

A sealed bundle records the rights state that existed *at capture time*. For a
historical native trace that state is ``UNKNOWN``, and it stays ``UNKNOWN``
forever: the bundle is immutable and rewriting it would destroy the historical
truth about what the trace itself contained.

An operator may nonetheless hold rights to that material and assert so. This
module evaluates such an assertion as an **external overlay**: a separate,
append-only ledger that binds to a specific sealed bundle by hash. Nothing here
reads, writes, or reinterprets the bundle.

Boundaries this module keeps deliberately:

* It does not define a rights policy. Admissibility comes from
  :data:`goldtrace_refinery.pack_record.ADMISSIBLE_RIGHTS_STATUS` and assertion
  validity from :func:`goldtrace_refinery.pack_record.validate_rights_assertion`,
  so Refinery never grows a second, divergent notion of acceptable rights.
* It grants at most ``RIGHTS_ASSERTED``. Rights evidence is not assay evidence,
  so it can never produce ``PASSED``.
* Every failure is closed and *visible*. An overlay the operator explicitly
  supplied but which cannot be parsed or evaluated is reported as an evaluated
  refusal, never as though no overlay had been requested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .pack_record import (
    ADMISSIBLE_RIGHTS_STATUS,
    rights_assertion_sha256,
    validate_rights_assertion,
)

#: The only disposition an overlay can grant. Never ``PASSED``.
RIGHTS_ASSERTED = "RIGHTS_ASSERTED"

LEDGER_SCHEMA = "gtdataworks.rights-ledger.v1"
LEDGER_SCHEMA_V2 = "gtdataworks.rights-ledger.v2"
ACCEPTED_LEDGER_SCHEMAS = frozenset({LEDGER_SCHEMA, LEDGER_SCHEMA_V2})

_ENTRY = "rights_entry"
_HEADER = "ledger_header"


class RightsOverlayError(ValueError):
    """The operator supplied an overlay that could not be loaded at all.

    Raised only for input the caller can fix (a missing file, unreadable JSON, a
    missing assertion-body sidecar). Evaluation outcomes — including every
    refusal — are returned as a :class:`RightsOverlayResult` instead, so a
    non-binding ledger still yields a receipt rather than an exception.
    """


@dataclass(frozen=True)
class RightsOverlayResult:
    """The outcome of evaluating an overlay against one sealed bundle."""

    evaluated: bool
    granted: bool
    reason: str
    ledger_path: str | None = None
    ledger_sha256: str | None = None
    ledger_schema: str | None = None
    bindings: dict[str, Any] = field(default_factory=dict)
    bindings_used: list[str] = field(default_factory=list)
    matched_entries: list[dict[str, Any]] = field(default_factory=list)
    rejected_entries: list[dict[str, Any]] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """Serialise for embedding in an ingot and for the overlay receipt."""
        return {
            "requested": True,
            "evaluated": self.evaluated,
            "granted": self.granted,
            "granted_status": RIGHTS_ASSERTED if self.granted else None,
            "reason": self.reason,
            "ledger_path": self.ledger_path,
            "ledger_sha256": self.ledger_sha256,
            "ledger_schema": self.ledger_schema,
            "bindings_required": self.bindings,
            # Which identifiers actually carried the match, so a reader never has
            # to assume a null-vs-null agreement counted for something.
            "bindings_used": self.bindings_used,
            "minimum_bindings": MINIMUM_BINDINGS,
            "matched_entries": self.matched_entries,
            "rejected_entries": self.rejected_entries,
        }


def _load_ledger_lines(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.is_file():
        raise RightsOverlayError(f"rights ledger not found: {ledger_path}")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RightsOverlayError(
                f"rights ledger line {number} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise RightsOverlayError(
                f"rights ledger line {number} is not a JSON object"
            )
        records.append(obj)
    if not records:
        raise RightsOverlayError(f"rights ledger is empty: {ledger_path}")
    return records


def _load_assertion_bodies(ledger_path: Path, entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Resolve assertion bodies and key them by their own canonical hash.

    Keying by recomputed hash — rather than by the hash the ledger claims — is
    what stops a body file being swapped underneath a ledger entry.
    """
    refs = {
        str(entry.get("assertion_body_ref"))
        for entry in entries
        if entry.get("assertion_body_ref")
    }
    bodies: dict[str, dict[str, Any]] = {}
    for ref in sorted(refs):
        candidate = (ledger_path.parent / Path(ref).name).resolve()
        if not candidate.is_file():
            raise RightsOverlayError(
                f"rights ledger references assertion bodies that are missing: {ref}"
            )
        try:
            doc = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RightsOverlayError(
                f"assertion body sidecar is not valid JSON: {candidate}"
            ) from exc
        assertions = doc.get("assertions") if isinstance(doc, dict) else None
        if not isinstance(assertions, dict):
            raise RightsOverlayError(
                f"assertion body sidecar has no 'assertions' object: {candidate}"
            )
        for block in assertions.values():
            assertion = (block or {}).get("rights_assertion")
            if not isinstance(assertion, dict):
                continue
            try:
                digest = rights_assertion_sha256(assertion)
            except (TypeError, ValueError, UnicodeError):
                continue
            bodies[digest] = assertion
    return bodies


def load_rights_overlay(ledger_path: Path | str) -> dict[str, Any]:
    """Load and structurally check a rights ledger. Raises on unusable input."""
    path = Path(ledger_path).resolve()
    records = _load_ledger_lines(path)
    header = next((r for r in records if r.get("record_type") == _HEADER), None)
    entries = [r for r in records if r.get("record_type") == _ENTRY]
    if not entries:
        raise RightsOverlayError(f"rights ledger contains no {_ENTRY} records: {path}")
    schema = (header or {}).get("ledger_schema") or (entries[0].get("ledger_schema"))
    if schema not in ACCEPTED_LEDGER_SCHEMAS:
        raise RightsOverlayError(
            f"unsupported rights ledger schema {schema!r}; "
            f"expected one of {sorted(ACCEPTED_LEDGER_SCHEMAS)}"
        )
    declared = (header or {}).get("entry_count")
    if isinstance(declared, int) and declared != len(entries):
        raise RightsOverlayError(
            f"rights ledger header declares {declared} entries but contains {len(entries)}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": schema,
        "header": header,
        "entries": entries,
        "assertion_bodies": _load_assertion_bodies(path, entries),
    }


#: The binding that must always match, and the ordered candidates for the second
#: independent binding. ``checkpoint_head`` is preferred when the bundle actually
#: carries one; ``event_head_sha256`` is the fallback because it is recomputed
#: from the sealed event chain rather than from the manifest, so it is a genuinely
#: separate identity rather than a restatement of the first.
PRIMARY_BINDING = "manifest_sha256"
SECOND_BINDING_CANDIDATES = ("checkpoint_head", "event_head_sha256")

MINIMUM_BINDINGS = 2


def _classify(entry_value: Any, bundle_value: Any) -> str:
    """MATCH / MISMATCH / UNAVAILABLE for one binding.

    ``UNAVAILABLE`` covers a null on either side. Two nulls are equal in Python
    but carry no identifying information, so null equality is never a binding.
    """
    if entry_value is None or bundle_value is None:
        return "UNAVAILABLE"
    return "MATCH" if entry_value == bundle_value else "MISMATCH"


def _bind_report(entry: dict[str, Any], bundle_ids: dict[str, Any]) -> dict[str, Any]:
    """Evaluate every candidate binding for one ledger entry."""
    seal = entry.get("canonical_seal")
    if not isinstance(seal, dict):
        return {"ok": False, "used": [], "reason": "entry has no canonical_seal object"}

    outcomes = {
        name: _classify(seal.get(name), bundle_ids.get(name))
        for name in (PRIMARY_BINDING, *SECOND_BINDING_CANDIDATES)
    }

    # A mismatch anywhere is positive evidence the entry describes something
    # else. It is never skipped in favour of a binding that happens to agree.
    mismatched = sorted(n for n, o in outcomes.items() if o == "MISMATCH")
    if mismatched:
        return {"ok": False, "used": [], "outcomes": outcomes,
                "reason": f"binding mismatch on {mismatched}"}

    if outcomes[PRIMARY_BINDING] != "MATCH":
        return {"ok": False, "used": [], "outcomes": outcomes,
                "reason": f"{PRIMARY_BINDING} is not available on both sides"}

    used = [PRIMARY_BINDING] + [
        n for n in SECOND_BINDING_CANDIDATES if outcomes[n] == "MATCH"
    ]
    if len(used) < MINIMUM_BINDINGS:
        return {
            "ok": False, "used": used, "outcomes": outcomes,
            "reason": (
                f"only {len(used)} non-null binding matched; {MINIMUM_BINDINGS} "
                "independently recomputed identifiers are required"
            ),
        }
    return {"ok": True, "used": used, "outcomes": outcomes, "reason": "bound"}


def evaluate_rights_overlay(
    overlay: dict[str, Any],
    *,
    manifest_sha256: str | None,
    checkpoint_head: Any,
    event_head_sha256: str | None = None,
) -> RightsOverlayResult:
    """Decide whether an overlay grants ``RIGHTS_ASSERTED`` for one bundle.

    Granting needs :data:`MINIMUM_BINDINGS` independently recomputed, non-null
    identifiers to agree. ``checkpoint_head`` is null on whole classes of sealed
    bundle, and ``None == None`` identifies nothing, so such a bundle must supply
    ``event_head_sha256`` as its second binding or it fails closed.
    """
    bundle_ids = {
        "manifest_sha256": manifest_sha256,
        "checkpoint_head": checkpoint_head,
        "event_head_sha256": event_head_sha256,
    }
    common = {
        "ledger_path": overlay.get("path"),
        "ledger_sha256": overlay.get("sha256"),
        "ledger_schema": overlay.get("schema"),
        "bindings": bundle_ids,
    }
    bodies: dict[str, dict[str, Any]] = overlay.get("assertion_bodies") or {}

    reports = [(e, _bind_report(e, bundle_ids)) for e in overlay["entries"]]
    matched = [e for e, r in reports if r["ok"]]
    bindings_used = sorted({b for _, r in reports if r["ok"] for b in r["used"]})
    if not matched:
        near = [r for _, r in reports if r.get("used")]
        detail = f"; closest entry: {near[0]['reason']}" if near else ""
        return RightsOverlayResult(
            evaluated=True,
            granted=False,
            reason=(
                f"no ledger entry bound this bundle on {MINIMUM_BINDINGS} independently "
                f"recomputed non-null identifiers{detail}"
            ),
            bindings_used=[],
            **common,
        )
    common["bindings_used"] = bindings_used

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for entry in matched:
        status = entry.get("rights_status")
        digest = entry.get("rights_assertion_sha256")
        summary = {
            "sealed_artifact_id": entry.get("sealed_artifact_id"),
            "rights_status": status,
            "rights_assertion_sha256": digest,
            "asserted_by": entry.get("asserted_by"),
            "asserted_at": entry.get("asserted_at"),
        }
        if status not in ADMISSIBLE_RIGHTS_STATUS:
            rejected.append({**summary, "rejection": f"rights_status {status!r} is not admissible"})
            continue
        assertion = bodies.get(digest) if isinstance(digest, str) else None
        if assertion is None:
            rejected.append(
                {**summary, "rejection": "no assertion body matches this assertion_sha256"}
            )
            continue
        errors = validate_rights_assertion(assertion)
        if errors:
            rejected.append({**summary, "rejection": f"assertion invalid: {errors}"})
            continue
        accepted.append(summary)

    # Two entries claiming the same rights_status with different assertions is a
    # genuine conflict. Several *distinct* classes over one bundle is not: mixed
    # material legitimately carries more than one class.
    by_status: dict[str, set[str]] = {}
    for item in accepted:
        by_status.setdefault(str(item["rights_status"]), set()).add(
            str(item["rights_assertion_sha256"])
        )
    conflicts = sorted(s for s, digests in by_status.items() if len(digests) > 1)
    if conflicts:
        return RightsOverlayResult(
            evaluated=True,
            granted=False,
            reason=(
                "ambiguous overlay: conflicting assertions for the same rights_status "
                f"{conflicts}"
            ),
            matched_entries=accepted,
            rejected_entries=rejected,
            **common,
        )

    if rejected:
        return RightsOverlayResult(
            evaluated=True,
            granted=False,
            reason=(
                f"{len(rejected)} of {len(matched)} binding entries failed validation; "
                "an overlay grants only when every binding entry is admissible and valid"
            ),
            matched_entries=accepted,
            rejected_entries=rejected,
            **common,
        )

    return RightsOverlayResult(
        evaluated=True,
        granted=True,
        reason=(
            f"{len(accepted)} binding entr{'y' if len(accepted) == 1 else 'ies'} matched on "
            f"{bindings_used}, carried an admissible rights_status, and passed assertion "
            "validation"
        ),
        matched_entries=accepted,
        rejected_entries=rejected,
        **common,
    )


def unevaluated(reason: str, ledger_path: Path | str | None) -> RightsOverlayResult:
    """An overlay was requested but could not be evaluated. Visible, not silent."""
    return RightsOverlayResult(
        evaluated=False,
        granted=False,
        reason=reason,
        ledger_path=str(ledger_path) if ledger_path is not None else None,
    )
