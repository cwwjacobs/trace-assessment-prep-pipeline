"""Private, structural operational telemetry derived from verified native traces.

This module creates one deliberately non-release product: an exact-inventory
HOLD pack containing only closed enums, numeric framing facts, and SHA-256
bindings already established by a sealed Labyrinth ``run.v7`` native import.
It never opens the source CAS itself and never emits source/run/session/native
identifiers, event identifiers, local/member paths, content, tool data, or
reasoning.

The pack is evidence, not a training-data lot.  UNKNOWN rights, HOLD, private
visibility, reasoning exclusion, and ``NOT_TRAINING_READY`` are immutable
contract values in every row and the manifest.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from .paths import ensure_labyrinth_importable, find_contract
from .release_privacy import (
    RELEASE_PRIVACY_RULESET_IDENTITY,
    sanitize_release_row,
    verify_release_privacy_receipt,
    verify_release_privacy_receipt_integrity,
)
from .verify_bundle import verify_mine_bundle


MANIFEST_SCHEMA_ID = "gtdataworks.native-operational-telemetry-manifest.v1"
ROW_SCHEMA_ID = "gtdataworks.native-operational-telemetry-row.v1"
LINEAGE_SCHEMA_ID = "gtdataworks.native-operational-telemetry-lineage.v1"
PRODUCT_KIND = "PRIVATE_NATIVE_OPERATIONAL_TELEMETRY_HOLD"
PRODUCT_VERSION = "1.0.0"
PROJECTION_RULESET_ID = "gtdataworks.native-operational-telemetry-projection.v1"

MANIFEST_CONTRACT = "native-telemetry-manifest.v1.schema.json"
ROW_CONTRACT = "native-telemetry-row.v1.schema.json"

_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_MAX_SMALL_FILE_BYTES = 4 * 1024 * 1024
_MAX_JSONL_FILE_BYTES = 256 * 1024 * 1024
_EXPECTED_ROOT_ENTRIES = frozenset(
    {"DATACARD.md", "MANIFEST.json", "provenance", "telemetry.jsonl"}
)
_EXPECTED_PROVENANCE_ENTRIES = frozenset(
    {"lineage.jsonl", "privacy-receipts.jsonl"}
)
_INVENTORIED_FILES = (
    "DATACARD.md",
    "provenance/lineage.jsonl",
    "provenance/privacy-receipts.jsonl",
    "telemetry.jsonl",
)

_GOVERNANCE = {
    "visibility": "PRIVATE",
    "rights_status": "UNKNOWN",
    "disposition": "HOLD",
    "release_admitted": False,
    "training_readiness": "NOT_TRAINING_READY",
    "reasoning_policy": "EXCLUDE",
}

PROJECTION_RULESET_DOCUMENT: dict[str, object] = {
    "identity": PROJECTION_RULESET_ID,
    "source_contract": "goldentrace.native-trace-index.v1",
    "row_contract": ROW_SCHEMA_ID,
    "allowed_source_fields": [
        "provider",
        "adapter",
        "adapter_spec_sha256",
        "record.index",
        "record.discriminator",
        "record.semantic_kind",
        "record.framing_status",
        "record.malformed",
        "record.malformed_reason",
        "record.oversize",
        "record.projection_available",
        "record.locator.framed_size_bytes",
        "record.locator.content_size_bytes",
        "record.locator.has_final_lf",
        "record.raw_sha256",
        "record.framed_sha256",
        "record.projection.authority",
        "event.event_hash",
        "source.source_sha256",
        "source.archive_sha256",
        "manifest.sha256",
        "event_head.sha256",
        "index.sha256",
    ],
    "excluded": [
        "content",
        "tool_data",
        "reasoning",
        "run_session_native_event_identifiers",
        "local_and_member_paths",
    ],
    "privacy_boundary": RELEASE_PRIVACY_RULESET_IDENTITY,
    "row_identity": "sha256-goldtrace-canonical-json-v1",
}


class NativeTelemetryPackError(ValueError):
    """A source or derived telemetry pack fails the closed contract."""


@dataclass(frozen=True)
class NativeTelemetryPackVerification:
    pack_path: Path
    provider: str
    adapter: str
    row_count: int
    manifest_sha256: str
    telemetry_sha256: str
    disposition: str = "HOLD"
    training_readiness: str = "NOT_TRAINING_READY"


@dataclass(frozen=True)
class NativeTelemetryPackExport:
    output_path: Path
    provider: str
    adapter: str
    row_count: int
    manifest_sha256: str
    telemetry_sha256: str


@dataclass(frozen=True)
class _SourceTruth:
    bundle_path: Path
    run_id: str
    provider: str
    adapter: str
    capture_mode: str
    adapter_spec_sha256: str
    manifest_sha256: str
    event_head_sha256: str
    index_sha256: str
    selected_source_sha256: str
    archive_sha256: str | None
    records: tuple[dict[str, object], ...]
    event_hash_by_id: Mapping[str, str]
    counts: Mapping[str, int]

    def stable_identity(self) -> tuple[object, ...]:
        return (
            self.run_id,
            self.provider,
            self.adapter,
            self.capture_mode,
            self.adapter_spec_sha256,
            self.manifest_sha256,
            self.event_head_sha256,
            self.index_sha256,
            self.selected_source_sha256,
            self.archive_sha256,
            tuple(
                (
                    row.get("index"),
                    row.get("raw_sha256"),
                    row.get("framed_sha256"),
                    (row.get("projection") or {}).get("event_id")
                    if isinstance(row.get("projection"), dict)
                    else None,
                )
                for row in self.records
            ),
        )


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise NativeTelemetryPackError("value is not canonical JSON") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256(_canonical_json(value))


PROJECTION_RULESET_SHA256 = _sha256_json(PROJECTION_RULESET_DOCUMENT)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise NativeTelemetryPackError("JSON object contains a duplicate key")
        value[key] = item
    return value


def _reject_constant(_value: str) -> object:
    raise NativeTelemetryPackError("JSON contains a non-finite number")


def _load_canonical_json_bytes(data: bytes, *, subject: str) -> dict[str, object]:
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, NativeTelemetryPackError) as exc:
        raise NativeTelemetryPackError(f"{subject} is not strict JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != data:
        raise NativeTelemetryPackError(f"{subject} is not a canonical JSON object")
    return value


def _parse_canonical_jsonl(data: bytes, *, subject: str) -> list[dict[str, object]]:
    if not data or not data.endswith(b"\n") or b"\r" in data:
        raise NativeTelemetryPackError(f"{subject} has invalid JSONL framing")
    rows: list[dict[str, object]] = []
    for line in data.splitlines():
        rows.append(_load_canonical_json_bytes(line, subject=f"{subject} record"))
    return rows


def _jsonl(rows: Iterable[dict[str, object]]) -> bytes:
    return b"".join(_canonical_json(row) + b"\n" for row in rows)


@lru_cache(maxsize=None)
def _load_schema(filename: str) -> dict[str, Any]:
    try:
        value = json.loads(find_contract(filename).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeTelemetryPackError("native telemetry contract is unavailable") from exc
    if not isinstance(value, dict):
        raise NativeTelemetryPackError("native telemetry contract is not an object")
    return value


@lru_cache(maxsize=None)
def _schema_validator(filename: str) -> Draft202012Validator:
    return Draft202012Validator(_load_schema(filename))


def _validate_schema(value: object, *, filename: str, subject: str) -> None:
    validator = _schema_validator(filename)
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    if errors:
        location = "/".join(str(part) for part in errors[0].path) or "<root>"
        raise NativeTelemetryPackError(f"{subject} violates its contract at {location}")


def _require(condition: object, message: str) -> None:
    if not condition:
        raise NativeTelemetryPackError(message)


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


@contextmanager
def _bound_directory(
    path: Path,
    *,
    subject: str = "native source",
    expected_mode: int | None = None,
) -> Iterable[Path]:
    """Hold and recheck a lexical directory binding for one complete operation."""

    candidate = Path(path).expanduser()
    try:
        initial = candidate.lstat()
    except OSError as exc:
        raise NativeTelemetryPackError(f"{subject} is unavailable") from exc

    def metadata_is_safe(metadata: os.stat_result) -> bool:
        return stat.S_ISDIR(metadata.st_mode) and (
            expected_mode is None or stat.S_IMODE(metadata.st_mode) == expected_mode
        )

    if stat.S_ISLNK(initial.st_mode) or not metadata_is_safe(initial):
        requirement = "a real directory" if expected_mode is None else "a real private directory"
        raise NativeTelemetryPackError(f"{subject} must be {requirement}")
    bound = Path(os.path.abspath(os.fspath(candidate)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(bound, flags)
    except OSError as exc:
        raise NativeTelemetryPackError(f"{subject} could not be bound") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not metadata_is_safe(opened)
            or _stat_identity(opened) != _stat_identity(initial)
        ):
            raise NativeTelemetryPackError(f"{subject} root changed while opening")
        yield bound
        after = os.fstat(descriptor)
        try:
            named = bound.lstat()
        except OSError as exc:
            raise NativeTelemetryPackError(f"{subject} root disappeared") from exc
        if (
            not metadata_is_safe(after)
            or stat.S_ISLNK(named.st_mode)
            or not metadata_is_safe(named)
            or _stat_identity(after) != _stat_identity(initial)
            or _stat_identity(named) != _stat_identity(initial)
        ):
            raise NativeTelemetryPackError(f"{subject} root binding changed")
    finally:
        os.close(descriptor)


def _source_truth(bundle_path: Path) -> _SourceTruth:
    with _bound_directory(bundle_path) as bundle:
        return _source_truth_from_bound_root(bundle)


def _source_truth_from_bound_root(bundle: Path) -> _SourceTruth:
    verification = verify_mine_bundle(bundle)
    if not verification.ok:
        raise NativeTelemetryPackError("native source bundle did not verify")
    _require(
        verification.seal_status == "SEALED_WITH_GAPS",
        "native source must be SEALED_WITH_GAPS",
    )
    details = verification.details or {}
    _require(
        details.get("classification") == "native_trace_import",
        "source is not a native-trace import",
    )
    _require(isinstance(verification.run_id, str), "verified source lacks run identity")
    _require(
        isinstance(verification.manifest_sha256, str)
        and _DIGEST.fullmatch(verification.manifest_sha256) is not None,
        "verified source lacks manifest binding",
    )

    ensure_labyrinth_importable()
    try:
        from goldentrace.capture.events import verify_event_stream
        from goldentrace.native_trace_contract import adapter_for_identity
    except Exception as exc:  # pragma: no cover - installation/configuration failure
        raise NativeTelemetryPackError("Labyrinth native contract is unavailable") from exc

    run_bytes = _read_regular_source_file(bundle / "run.json", max_bytes=_MAX_SMALL_FILE_BYTES)
    index_bytes = _read_regular_source_file(
        bundle / "native-trace-index.json", max_bytes=_MAX_JSONL_FILE_BYTES
    )
    run = _load_canonical_json_bytes(run_bytes, subject="native source run")
    index = _load_canonical_json_bytes(index_bytes, subject="native source index")
    run_source = run.get("run_source")
    _require(
        run.get("schema_version") == "goldentrace.run.v7"
        and run.get("classification") == "native_trace_import"
        and run.get("import_status") == "COMPLETED"
        and run.get("verification_status") == "VERIFIED_WITH_GAPS"
        and run.get("seal_status") == "SEALED_WITH_GAPS"
        and isinstance(run_source, dict),
        "source run is not an exact completed native run.v7 import",
    )
    assert isinstance(run_source, dict)
    try:
        spec = adapter_for_identity(
            provider=run_source.get("actor"),
            adapter_id=run_source.get("adapter"),
            capture_mode=run_source.get("capture_mode"),
            adapter_spec_sha256=run_source.get("adapter_spec_sha256"),
        )
    except Exception as exc:
        raise NativeTelemetryPackError("source native adapter identity is unregistered") from exc

    index_sha256 = _sha256(index_bytes)
    _require(
        index.get("schema_version") == "goldentrace.native-trace-index.v1"
        and index.get("classification") == "native_trace_import"
        and index.get("provider") == spec.provider
        and index.get("adapter") == spec.adapter_id
        and index.get("capture_mode") == spec.capture_mode
        and index.get("adapter_spec_sha256") == spec.spec_sha256
        and run.get("native_trace_index")
        == {
            "schema_version": "goldentrace.native-trace-index.v1",
            "path": "native-trace-index.json",
            "sha256": index_sha256,
        },
        "native source index identity is not bound to the registered adapter",
    )
    governance = index.get("governance")
    _require(
        isinstance(governance, dict)
        and governance.get("rights")
        == {"status": "UNKNOWN", "basis": "operator_review_required"}
        and governance.get("disposition")
        == {
            "status": "HOLD",
            "readiness": "not_training_ready",
            "reason": "historical_native_trace_not_semantically_or_rights_reviewed",
        }
        and governance.get("model_reasoning")
        == {
            "default": "EXCLUDE",
            "reason": "native_model_reasoning_not_admitted_by_default",
        },
        "native source governance is not fail closed",
    )

    source = index.get("source")
    records = index.get("records")
    _require(isinstance(source, dict), "native source descriptor is malformed")
    _require(isinstance(records, list) and records, "native source index has no records")
    assert isinstance(source, dict) and isinstance(records, list)
    source_sha256 = source.get("source_sha256")
    archive_sha256 = source.get("archive_sha256")
    _require(
        isinstance(source_sha256, str) and _DIGEST.fullmatch(source_sha256) is not None,
        "native source bytes lack an exact SHA-256 binding",
    )
    _require(
        archive_sha256 is None
        or (isinstance(archive_sha256, str) and _DIGEST.fullmatch(archive_sha256) is not None),
        "native archive binding is malformed",
    )

    try:
        events = verify_event_stream(
            bundle / "events" / "events.ndjson",
            expected_run_id=verification.run_id,
            expected_classification="native_trace_import",
        )
    except Exception as exc:
        raise NativeTelemetryPackError("native event stream did not verify") from exc
    _require(bool(events), "native event stream is empty")
    event_head = getattr(events[-1], "event_hash", None)
    _require(
        isinstance(event_head, str)
        and event_head == details.get("event_head_sha256"),
        "native event head differs from sealed verification",
    )
    event_hash_by_id = {
        str(getattr(event, "event_id")): str(getattr(event, "event_hash"))
        for event in events
    }

    normalized_records: list[dict[str, object]] = []
    for expected_ordinal, record in enumerate(records, start=1):
        _require(isinstance(record, dict), "native source record is malformed")
        assert isinstance(record, dict)
        projection = record.get("projection")
        _require(
            record.get("index") == expected_ordinal and isinstance(projection, dict),
            "native source record ordering/projection is malformed",
        )
        assert isinstance(projection, dict)
        event_id = projection.get("event_id")
        _require(
            isinstance(event_id, str) and event_id in event_hash_by_id,
            "native source projection is not bound to a verified event",
        )
        normalized_records.append(record)

    count_fields = (
        "record_count",
        "known_record_count",
        "unknown_record_count",
        "malformed_record_count",
        "oversize_record_count",
    )
    counts: dict[str, int] = {}
    for field in count_fields:
        value = index.get(field)
        _require(type(value) is int and value >= 0, "native source counts are malformed")
        counts[field] = value
    _require(counts["record_count"] == len(normalized_records), "native record count differs")

    return _SourceTruth(
        bundle_path=bundle,
        run_id=verification.run_id,
        provider=spec.provider,
        adapter=spec.adapter_id,
        capture_mode=spec.capture_mode,
        adapter_spec_sha256=spec.spec_sha256,
        manifest_sha256=verification.manifest_sha256,
        event_head_sha256=event_head,
        index_sha256=index_sha256,
        selected_source_sha256=source_sha256,
        archive_sha256=archive_sha256,
        records=tuple(normalized_records),
        event_hash_by_id=event_hash_by_id,
        counts=counts,
    )


def _read_bound_regular_file(
    path: Path,
    *,
    max_bytes: int,
    expected_mode: int | None,
    unavailable: str,
    unsafe: str,
) -> bytes:
    """Read one exact regular file while holding and rechecking its descriptor."""

    candidate = Path(path)
    try:
        initial = candidate.lstat()
    except OSError as exc:
        raise NativeTelemetryPackError(unavailable) from exc

    def metadata_is_safe(metadata: os.stat_result) -> bool:
        return (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 1
            and 0 <= metadata.st_size <= max_bytes
            and (
                expected_mode is None
                or stat.S_IMODE(metadata.st_mode) == expected_mode
            )
        )

    if stat.S_ISLNK(initial.st_mode) or not metadata_is_safe(initial):
        raise NativeTelemetryPackError(unsafe)

    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise NativeTelemetryPackError(unsafe) from exc

    try:
        opened = os.fstat(descriptor)
        if (
            not metadata_is_safe(opened)
            or _stat_identity(opened) != _stat_identity(initial)
        ):
            raise NativeTelemetryPackError(unsafe)

        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            try:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            except OSError as exc:
                raise NativeTelemetryPackError(unsafe) from exc
            if not chunk:
                raise NativeTelemetryPackError(unsafe)
            chunks.append(chunk)
            remaining -= len(chunk)
        try:
            trailing = os.read(descriptor, 1)
            after = os.fstat(descriptor)
        except OSError as exc:
            raise NativeTelemetryPackError(unsafe) from exc
        if trailing or _stat_identity(after) != _stat_identity(opened):
            raise NativeTelemetryPackError(unsafe)
    finally:
        os.close(descriptor)

    try:
        named = candidate.lstat()
    except OSError as exc:
        raise NativeTelemetryPackError(unsafe) from exc
    if (
        stat.S_ISLNK(named.st_mode)
        or not metadata_is_safe(named)
        or _stat_identity(named) != _stat_identity(initial)
    ):
        raise NativeTelemetryPackError(unsafe)
    return b"".join(chunks)


def _read_regular_source_file(path: Path, *, max_bytes: int) -> bytes:
    return _read_bound_regular_file(
        path,
        max_bytes=max_bytes,
        expected_mode=None,
        unavailable="required native source file is unavailable",
        unsafe="required native source file is unsafe",
    )


def _source_view(truth: _SourceTruth) -> dict[str, object]:
    return {
        "provider": truth.provider,
        "adapter": truth.adapter,
        "capture_mode": truth.capture_mode,
        "adapter_spec_sha256": truth.adapter_spec_sha256,
        "source_manifest_sha256": truth.manifest_sha256,
        "event_head_sha256": truth.event_head_sha256,
        "native_trace_index_sha256": truth.index_sha256,
        "selected_source_sha256": truth.selected_source_sha256,
        "archive_sha256": truth.archive_sha256,
    }


def _row_source_binding(
    truth: _SourceTruth,
    record: Mapping[str, object],
) -> dict[str, object]:
    projection = record.get("projection")
    _require(isinstance(projection, dict), "native projection is malformed")
    assert isinstance(projection, dict)
    event_id = projection.get("event_id")
    _require(isinstance(event_id, str), "native projection lacks an event binding")
    event_hash = truth.event_hash_by_id.get(event_id)
    _require(isinstance(event_hash, str), "native projection event is unverified")
    return {
        "source_manifest_sha256": truth.manifest_sha256,
        "event_head_sha256": truth.event_head_sha256,
        "native_trace_index_sha256": truth.index_sha256,
        "selected_source_sha256": truth.selected_source_sha256,
        "archive_sha256": truth.archive_sha256,
        "record_raw_sha256": record.get("raw_sha256"),
        "record_framed_sha256": record.get("framed_sha256"),
        "projection_event_sha256": event_hash,
    }


def _row_identity(
    truth: _SourceTruth,
    record: Mapping[str, object],
) -> str:
    return "ntelem-" + _sha256_json(
        {
            "lane": "operational_telemetry",
            "adapter": truth.adapter,
            "source_manifest_sha256": truth.manifest_sha256,
            "native_trace_index_sha256": truth.index_sha256,
            "record_raw_sha256": record.get("raw_sha256"),
            "record_ordinal": record.get("index"),
            "projection_ruleset_sha256": PROJECTION_RULESET_SHA256,
        }
    )


def _project_row(truth: _SourceTruth, record: Mapping[str, object]) -> dict[str, object]:
    locator = record.get("locator")
    projection = record.get("projection")
    _require(isinstance(locator, dict), "native record locator is malformed")
    _require(isinstance(projection, dict), "native record projection is malformed")
    assert isinstance(locator, dict) and isinstance(projection, dict)
    row = {
        "schema_id": ROW_SCHEMA_ID,
        "row_id": _row_identity(truth, record),
        "provider": truth.provider,
        "adapter": truth.adapter,
        "adapter_spec_sha256": truth.adapter_spec_sha256,
        "record_ordinal": record.get("index"),
        "semantic_kind": record.get("semantic_kind"),
        "supported_discriminator": record.get("discriminator"),
        "framing_status": record.get("framing_status"),
        "malformed": record.get("malformed"),
        "malformed_reason": record.get("malformed_reason"),
        "oversize": record.get("oversize"),
        "projection_available": record.get("projection_available"),
        "framed_size_bytes": locator.get("framed_size_bytes"),
        "content_size_bytes": locator.get("content_size_bytes"),
        "has_final_lf": locator.get("has_final_lf"),
        "authority": projection.get("authority"),
        "semantic_status": "NATIVE_TELEMETRY_ONLY_NOT_EXECUTION_WITNESS",
        "source_binding": _row_source_binding(truth, record),
        "governance": dict(_GOVERNANCE),
    }
    _validate_schema(row, filename=ROW_CONTRACT, subject="projected telemetry row")
    return row


def _derive_documents(
    truth: _SourceTruth,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    lineage: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    for record in truth.records:
        row = _project_row(truth, record)
        row_id = row["row_id"]
        assert isinstance(row_id, str)
        sanitized, receipt = sanitize_release_row(row, record_id=row_id)
        if sanitized != row or receipt.get("status") != "PASS_AUTOMATED_SCAN":
            raise NativeTelemetryPackError(
                "closed structural telemetry unexpectedly required privacy redaction"
            )
        errors = verify_release_privacy_receipt(receipt, sanitized)
        if errors:
            raise NativeTelemetryPackError("generated privacy receipt did not verify")
        _validate_schema(sanitized, filename=ROW_CONTRACT, subject="privacy-treated row")
        receipt_exact_sha256 = _sha256_json(receipt)
        rows.append(sanitized)
        receipts.append(receipt)
        lineage.append(
            {
                "schema_id": LINEAGE_SCHEMA_ID,
                "row_id": row_id,
                "released_row_sha256": receipt["output_row_sha256"],
                "privacy_receipt_sha256": receipt_exact_sha256,
                "privacy_ruleset_sha256": receipt["ruleset_sha256"],
                "source_binding": dict(sanitized["source_binding"]),
            }
        )
    return rows, lineage, receipts


def _data_card() -> bytes:
    return (
        "# Private native operational telemetry HOLD pack\n\n"
        "This pack is a structural evidence derivative, not a training dataset or ProductLot.\n\n"
        "- Visibility: `PRIVATE`\n"
        "- Rights: `UNKNOWN`\n"
        "- Disposition: `HOLD`\n"
        "- Release admitted: `false`\n"
        "- Training readiness: `NOT_TRAINING_READY`\n"
        "- Reasoning policy: `EXCLUDE`\n\n"
        "Rows contain only closed adapter/semantic enums, numeric framing facts, and "
        "SHA-256 provenance bindings. They exclude source content, tool data, reasoning "
        "material, source/run/session/native/event identifiers, and local or member paths. "
        "Tool-related rows are provider testimony, not execution or filesystem witnesses.\n"
    ).encode("utf-8")


def _manifest(
    truth: _SourceTruth,
    payloads: Mapping[str, bytes],
    *,
    row_count: int,
) -> dict[str, object]:
    files = [
        {"name": name, "size_bytes": len(payloads[name]), "sha256": _sha256(payloads[name])}
        for name in _INVENTORIED_FILES
    ]
    counts = {
        "row_count": row_count,
        "known_record_count": truth.counts["known_record_count"],
        "unknown_record_count": truth.counts["unknown_record_count"],
        "malformed_record_count": truth.counts["malformed_record_count"],
        "oversize_record_count": truth.counts["oversize_record_count"],
        "privacy_receipt_count": row_count,
    }
    manifest = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "product_kind": PRODUCT_KIND,
        "product_version": PRODUCT_VERSION,
        "source": _source_view(truth),
        "projection_ruleset": {
            "identity": PROJECTION_RULESET_ID,
            "sha256": PROJECTION_RULESET_SHA256,
        },
        "governance": dict(_GOVERNANCE),
        "counts": counts,
        "files": files,
    }
    _validate_schema(manifest, filename=MANIFEST_CONTRACT, subject="telemetry manifest")
    return manifest


def _safe_output(path: Path) -> tuple[Path, Path]:
    raw = Path(path).expanduser()
    if raw.exists() or raw.is_symlink():
        raise NativeTelemetryPackError("native telemetry output already exists")
    raw_parent = raw.parent
    raw_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_metadata = raw_parent.lstat()
    if not stat.S_ISDIR(parent_metadata.st_mode) or stat.S_ISLNK(parent_metadata.st_mode):
        raise NativeTelemetryPackError("native telemetry output parent is unsafe")
    output = Path(os.path.abspath(os.fspath(raw)))
    return output, output.parent


def _write_private_file(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600, follow_symlinks=False)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> bool:
    """Use the Linux atomic primitive, returning false when the filesystem lacks it."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:  # pragma: no cover - this Build runs on Linux/glibc
        return False
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise NativeTelemetryPackError("native telemetry output already exists")
    unsupported = {
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        getattr(errno, "ENOTSUP", errno.EINVAL),
    }
    if error in unsupported:
        return False
    raise NativeTelemetryPackError("atomic native telemetry installation failed") from OSError(
        error, os.strerror(error)
    )


def _install_directory_noreplace(source: Path, destination: Path) -> None:
    """Install without overwrite; make unsupported-filesystem fallback fail closed.

    Linux filesystems that support ``RENAME_NOREPLACE`` get one atomic rename.
    eCryptfs rejects that flag. There we atomically reserve the absent target
    with ``mkdir``, retain an extra ``.INSTALLING`` entry while moving the
    already-fsynced tree, and remove that marker only after the final fsync.
    A concurrent verifier therefore sees either no target, an unverifiable
    incomplete target, or the complete exact inventory. A crash can leave a
    private incomplete directory that subsequent producers refuse to replace.
    """

    if _rename_directory_noreplace(source, destination):
        _fsync_directory(destination.parent)
        return

    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise NativeTelemetryPackError("native telemetry output already exists") from exc
    marker = destination / ".INSTALLING"
    _write_private_file(marker, b"native telemetry installation incomplete\n")
    try:
        for entry in sorted(os.scandir(source), key=lambda item: item.name):
            os.rename(entry.path, destination / entry.name)
        source.rmdir()
        provenance = destination / "provenance"
        if provenance.is_dir():
            _fsync_directory(provenance)
        _fsync_directory(destination)
        marker.unlink()
        _fsync_directory(destination)
    except BaseException as exc:
        raise NativeTelemetryPackError(
            "native telemetry installation is incomplete and remains fail closed"
        ) from exc
    _fsync_directory(destination.parent)


def produce_native_telemetry_pack(
    source_bundle: Path,
    output_path: Path,
) -> NativeTelemetryPackExport:
    """Derive and atomically install one private, non-overwriting HOLD pack."""

    first_truth = _source_truth(source_bundle)
    rows, lineage, receipts = _derive_documents(first_truth)
    payloads = {
        "DATACARD.md": _data_card(),
        "provenance/lineage.jsonl": _jsonl(lineage),
        "provenance/privacy-receipts.jsonl": _jsonl(receipts),
        "telemetry.jsonl": _jsonl(rows),
    }
    manifest = _manifest(first_truth, payloads, row_count=len(rows))
    manifest_bytes = _canonical_json(manifest)

    second_truth = _source_truth(source_bundle)
    if first_truth.stable_identity() != second_truth.stable_identity():
        raise NativeTelemetryPackError("native source changed during telemetry derivation")

    output, parent = _safe_output(output_path)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    os.chmod(temporary, 0o700)
    installed = False
    try:
        provenance = temporary / "provenance"
        provenance.mkdir(mode=0o700)
        for relative, data in payloads.items():
            _write_private_file(temporary / relative, data)
        _write_private_file(temporary / "MANIFEST.json", manifest_bytes)
        _fsync_directory(provenance)
        _fsync_directory(temporary)

        verify_native_telemetry_pack(temporary, source_bundle=source_bundle)
        _install_directory_noreplace(temporary, output)
        installed = True
        verification = verify_native_telemetry_pack(output, source_bundle=source_bundle)
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)

    return NativeTelemetryPackExport(
        output_path=output,
        provider=verification.provider,
        adapter=verification.adapter,
        row_count=verification.row_count,
        manifest_sha256=verification.manifest_sha256,
        telemetry_sha256=verification.telemetry_sha256,
    )


def _read_pack_file(root: Path, relative: str, *, max_bytes: int) -> bytes:
    return _read_bound_regular_file(
        root / relative,
        max_bytes=max_bytes,
        expected_mode=0o600,
        unavailable="native telemetry pack file is unavailable",
        unsafe="native telemetry pack file is unsafe",
    )


def _verify_pack_topology(root: Path) -> None:
    observed_root = {entry.name for entry in os.scandir(root)}
    if observed_root != _EXPECTED_ROOT_ENTRIES:
        raise NativeTelemetryPackError("native telemetry pack exact root inventory differs")
    provenance = root / "provenance"
    metadata = provenance.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise NativeTelemetryPackError("native telemetry provenance directory is unsafe")
    observed_provenance = {entry.name for entry in os.scandir(provenance)}
    if observed_provenance != _EXPECTED_PROVENANCE_ENTRIES:
        raise NativeTelemetryPackError("native telemetry provenance inventory differs")


def _verify_lineage_shape(lineage: Mapping[str, object]) -> None:
    expected = {
        "schema_id",
        "row_id",
        "released_row_sha256",
        "privacy_receipt_sha256",
        "privacy_ruleset_sha256",
        "source_binding",
    }
    if set(lineage) != expected:
        raise NativeTelemetryPackError("native telemetry lineage fields differ")
    if lineage.get("schema_id") != LINEAGE_SCHEMA_ID:
        raise NativeTelemetryPackError("native telemetry lineage schema differs")
    for field in (
        "released_row_sha256",
        "privacy_receipt_sha256",
        "privacy_ruleset_sha256",
    ):
        value = lineage.get(field)
        if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
            raise NativeTelemetryPackError("native telemetry lineage hash is malformed")


def _verify_native_telemetry_pack_bound(
    root: Path,
    *,
    first_truth: _SourceTruth,
) -> NativeTelemetryPackVerification:
    _verify_pack_topology(root)

    manifest_bytes = _read_pack_file(root, "MANIFEST.json", max_bytes=_MAX_SMALL_FILE_BYTES)
    data_card = _read_pack_file(root, "DATACARD.md", max_bytes=_MAX_SMALL_FILE_BYTES)
    telemetry_bytes = _read_pack_file(
        root, "telemetry.jsonl", max_bytes=_MAX_JSONL_FILE_BYTES
    )
    lineage_bytes = _read_pack_file(
        root, "provenance/lineage.jsonl", max_bytes=_MAX_JSONL_FILE_BYTES
    )
    receipt_bytes = _read_pack_file(
        root, "provenance/privacy-receipts.jsonl", max_bytes=_MAX_JSONL_FILE_BYTES
    )
    if data_card != _data_card():
        raise NativeTelemetryPackError("native telemetry data card differs from v1")

    manifest = _load_canonical_json_bytes(manifest_bytes, subject="telemetry manifest")
    _validate_schema(manifest, filename=MANIFEST_CONTRACT, subject="telemetry manifest")
    actual_payloads = {
        "DATACARD.md": data_card,
        "provenance/lineage.jsonl": lineage_bytes,
        "provenance/privacy-receipts.jsonl": receipt_bytes,
        "telemetry.jsonl": telemetry_bytes,
    }
    files = manifest.get("files")
    _require(isinstance(files, list), "telemetry manifest inventory is malformed")
    expected_inventory = [
        {
            "name": name,
            "size_bytes": len(actual_payloads[name]),
            "sha256": _sha256(actual_payloads[name]),
        }
        for name in _INVENTORIED_FILES
    ]
    if files != expected_inventory:
        raise NativeTelemetryPackError("telemetry manifest does not bind exact pack files")

    rows = _parse_canonical_jsonl(telemetry_bytes, subject="telemetry.jsonl")
    lineage = _parse_canonical_jsonl(lineage_bytes, subject="lineage.jsonl")
    receipts = _parse_canonical_jsonl(receipt_bytes, subject="privacy-receipts.jsonl")
    if not (len(rows) == len(lineage) == len(receipts) == len(first_truth.records)):
        raise NativeTelemetryPackError("telemetry pack record ledgers do not reconcile")
    row_ids = [row.get("row_id") for row in rows]
    if not all(isinstance(row_id, str) for row_id in row_ids) or len(row_ids) != len(set(row_ids)):
        raise NativeTelemetryPackError("telemetry row identities are missing or duplicated")

    # Re-project source truth, but do not regenerate historical privacy
    # receipts under the current policy. Receipt integrity is dispatched by
    # the policy tuple carried in each exact receipt below.
    expected_rows = [_project_row(first_truth, record) for record in first_truth.records]
    for actual, expected in zip(rows, expected_rows, strict=True):
        _validate_schema(actual, filename=ROW_CONTRACT, subject="telemetry row")
        if actual != expected:
            raise NativeTelemetryPackError("telemetry row differs from verified source projection")
    for row, actual_lineage, actual_receipt in zip(rows, lineage, receipts, strict=True):
        _verify_lineage_shape(actual_lineage)
        row_id = row.get("row_id")
        assert isinstance(row_id, str)
        receipt_verification = verify_release_privacy_receipt_integrity(
            actual_receipt,
            row,
            record_id=row_id,
        )
        if not receipt_verification.ok:
            raise NativeTelemetryPackError("telemetry privacy receipt differs from its exact row")
        # This projection is intentionally closed and must never require
        # redaction. Historical v1 is accepted only as integrity evidence for
        # an already-private HOLD pack; current production remains v2.
        if (
            actual_receipt.get("output_changed") is not False
            or actual_receipt.get("findings_count") != 0
            or actual_receipt.get("input_row_sha256") != actual_receipt.get("output_row_sha256")
        ):
            raise NativeTelemetryPackError("telemetry privacy receipt claims row redaction")
        expected_lineage = {
            "schema_id": LINEAGE_SCHEMA_ID,
            "row_id": row_id,
            "released_row_sha256": actual_receipt.get("output_row_sha256"),
            "privacy_receipt_sha256": _sha256_json(actual_receipt),
            "privacy_ruleset_sha256": actual_receipt.get("ruleset_sha256"),
            "source_binding": dict(row["source_binding"]),
        }
        if actual_lineage != expected_lineage:
            raise NativeTelemetryPackError(
                "telemetry lineage differs from verified source projection"
            )

    expected_manifest = _manifest(first_truth, actual_payloads, row_count=len(rows))
    if manifest != expected_manifest:
        raise NativeTelemetryPackError("telemetry manifest widens or differs from source truth")

    return NativeTelemetryPackVerification(
        pack_path=root,
        provider=first_truth.provider,
        adapter=first_truth.adapter,
        row_count=len(rows),
        manifest_sha256=_sha256(manifest_bytes),
        telemetry_sha256=_sha256(telemetry_bytes),
    )


def verify_native_telemetry_pack(
    pack_path: Path,
    *,
    source_bundle: Path,
) -> NativeTelemetryPackVerification:
    """Verify exact pack bytes and independently rebind every row to its source."""

    first_truth = _source_truth(source_bundle)
    with _bound_directory(
        pack_path,
        subject="native telemetry pack",
        expected_mode=0o700,
    ) as root:
        verification = _verify_native_telemetry_pack_bound(
            root,
            first_truth=first_truth,
        )
        second_truth = _source_truth(source_bundle)
        if first_truth.stable_identity() != second_truth.stable_identity():
            raise NativeTelemetryPackError(
                "native source changed during telemetry verification"
            )

    return verification


__all__ = [
    "LINEAGE_SCHEMA_ID",
    "MANIFEST_CONTRACT",
    "MANIFEST_SCHEMA_ID",
    "NativeTelemetryPackError",
    "NativeTelemetryPackExport",
    "NativeTelemetryPackVerification",
    "PRODUCT_KIND",
    "PROJECTION_RULESET_DOCUMENT",
    "PROJECTION_RULESET_ID",
    "PROJECTION_RULESET_SHA256",
    "ROW_CONTRACT",
    "ROW_SCHEMA_ID",
    "produce_native_telemetry_pack",
    "verify_native_telemetry_pack",
]
