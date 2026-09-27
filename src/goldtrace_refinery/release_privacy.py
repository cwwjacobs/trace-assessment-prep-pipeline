"""Deterministic privacy gate for an exact model-facing JSON row.

Casting scans an evidence stream, but a product row can be assembled from
additional fields after that scan.  This module is the final boundary: it
sanitizes the fully assembled row, binds the exact canonical input and output
bytes, and emits a receipt that contains counts only -- never matched text,
previews, or JSON paths.

The receipt is an integrity record, not an authenticity signature.  A caller
must still bind ``input_row_sha256`` to the upstream record/provenance chain.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from goldtrace_mint.factory.sanitizer import (
    DEFAULT_REDACTION_TOKENS,
    PIICategory,
    PIIFinding,
    PIISanitizer,
)

__all__ = [
    "RELEASE_PRIVACY_RECEIPT_SCHEMA",
    "RELEASE_PRIVACY_RECEIPT_SCHEMA_V1",
    "RELEASE_PRIVACY_RULESET_IDENTITY",
    "RELEASE_PRIVACY_RULESET_IDENTITY_V1",
    "ReleasePrivacyVerification",
    "ReleasePrivacyError",
    "release_ruleset_sha256",
    "sanitize_release_row",
    "verify_release_privacy_receipt",
    "verify_release_privacy_receipt_integrity",
]


RELEASE_PRIVACY_RECEIPT_SCHEMA_V1 = "goldtrace.release-privacy-receipt.v1"
RELEASE_PRIVACY_RULESET_IDENTITY_V1 = "goldtrace.release-privacy-ruleset.v1"
RELEASE_PRIVACY_RECEIPT_SCHEMA = "goldtrace.release-privacy-receipt.v2"
RELEASE_PRIVACY_RULESET_IDENTITY = "goldtrace.release-privacy-ruleset.v2"

_PASS_SCAN = "PASS_AUTOMATED_SCAN"
_PASS_REDACTION = "PASS_AUTOMATED_REDACTION"
_CURRENT_POLICY_VERIFIED = "CURRENT_POLICY_VERIFIED"
_HISTORICAL_INTEGRITY_ONLY = "HISTORICAL_INTEGRITY_ONLY"
_INVALID = "INVALID"
_MAX_JSON_DEPTH = 128
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX64_CANDIDATE_RE = re.compile(r"^[0-9A-Fa-f]{64}$")
_ENTROPY_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9_+/-])[A-Za-z0-9_+/-]{24,}={0,2}(?![A-Za-z0-9_+/=-])"
)
_ENTROPY_MIN_LENGTH = 24
_ENTROPY_MIN_BITS_PER_CHARACTER = 4.2
_ENTROPY_MIN_CHARACTER_CLASSES = 3
_HEX64_MIN_BITS_PER_CHARACTER = 3.5
_STRUCTURED_SHA256_EXACT_FIELDS = frozenset(
    {
        # Generic manifest entries deliberately expose this exact field.
        "sha256",
        # Closed native-telemetry row contract fields.  The allowance is an
        # explicit schema surface, not a caller-chosen ``*_sha256`` suffix.
        "adapter_spec_sha256",
        "archive_sha256",
        "event_head_sha256",
        "native_trace_index_sha256",
        "projection_event_sha256",
        "record_framed_sha256",
        "record_raw_sha256",
        "selected_source_sha256",
        "source_manifest_sha256",
    }
)
_STRUCTURED_PREFIXED_SHA256_FIELDS = {"row_id": "ntelem-"}
_COVERAGE_SCOPE = "all_json_string_values_and_object_keys"
_COVERAGE_DETECTORS = (
    "pii-sanitizer-patterns-v1",
    "sensitive-phrase-registry-v1",
    "workspace-location-v1",
    "high-entropy-token-v1",
)
_COVERAGE_PHASE_FIELDS = frozenset(
    {
        "object_key_strings_total",
        "object_key_strings_scanned",
        "string_values_total",
        "string_values_scanned",
        "utf8_bytes_total",
        "utf8_bytes_scanned",
        "complete",
    }
)
# Release receipts must be verifiable on a machine whose login/home differs
# from the producer's.  The promoted sanitizer accepts an operator home so it
# can preserve that operator's path structure, but that makes its compiled
# ruleset host-specific.  At this publication boundary we instead use a stable
# sentinel: real /home/<user> and /Users/<user> paths take the generic opaque
# redaction path, while producer and verifier instantiate byte-identical rules.
_PORTABLE_RELEASE_WORKSPACE_HOME = "/__gtdataworks_release_workspace__"
_RECEIPT_FIELDS_V1 = frozenset(
    {
        "schema_id",
        "record_id_sha256",
        "ruleset_identity",
        "ruleset_sha256",
        "input_row_sha256",
        "output_row_sha256",
        "findings_count",
        "findings_by_category",
        "output_changed",
        "residual_findings_count",
        "residual_findings_by_category",
        "status",
        "receipt_sha256",
    }
)
_RECEIPT_FIELDS_V2 = _RECEIPT_FIELDS_V1 | {"coverage"}

# This is the exact portable v1 ruleset hash emitted before the v2 entropy and
# coverage policy existed.  It is retained only so historical evidence can be
# checked as integrity evidence.  It is never sufficient for current release
# admission.
_PORTABLE_V1_RULESET_SHA256 = "080995bf28e4237797b87807b92d0d1fd338dd66e2a3bb2539360c300c67e094"


class ReleasePrivacyError(ValueError):
    """The row cannot safely cross the final privacy boundary."""


@dataclass(frozen=True)
class ReleasePrivacyVerification:
    """Version-aware receipt result with an explicit admission boundary."""

    status: str
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _canonical_json_bytes(value: Any) -> bytes:
    """Return the exact bytes used by every release-row and receipt hash."""

    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return rendered.encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ReleasePrivacyError(f"value is not canonical JSON: {exc}") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    try:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    except UnicodeError as exc:
        raise ReleasePrivacyError("record_id is not valid UTF-8 text") from exc


def _validate_json_tree(
    value: Any,
    *,
    path: str = "$",
    depth: int = 0,
    active_containers: set[int] | None = None,
) -> None:
    """Reject ambiguous/non-JSON values before the sanitizer traverses them."""

    if depth > _MAX_JSON_DEPTH:
        raise ReleasePrivacyError(f"JSON row exceeds maximum depth at {path}")

    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ReleasePrivacyError(f"non-finite number at {path}")
        return

    if active_containers is None:
        active_containers = set()

    if type(value) is dict:
        marker = id(value)
        if marker in active_containers:
            raise ReleasePrivacyError(f"cyclic JSON object at {path}")
        active_containers.add(marker)
        try:
            for index, (key, child) in enumerate(value.items()):
                if type(key) is not str:
                    raise ReleasePrivacyError(f"object key at {path} must be a string")
                _validate_json_tree(
                    child,
                    # Paths are deliberately structural: an object key may
                    # itself contain the secret that made the row malformed.
                    path=f"{path}.value[{index}]",
                    depth=depth + 1,
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(marker)
        return

    if type(value) is list:
        marker = id(value)
        if marker in active_containers:
            raise ReleasePrivacyError(f"cyclic JSON array at {path}")
        active_containers.add(marker)
        try:
            for index, child in enumerate(value):
                _validate_json_tree(
                    child,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(marker)
        return

    raise ReleasePrivacyError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _normalized_workspace_home(workspace_home: str | None) -> str:
    if workspace_home is None:
        workspace_home = _PORTABLE_RELEASE_WORKSPACE_HOME
    if type(workspace_home) is not str or not workspace_home:
        raise ReleasePrivacyError("workspace_home must be a non-empty absolute path")
    normalized = workspace_home.rstrip("/")
    if not normalized or not normalized.startswith("/") or "\x00" in normalized:
        raise ReleasePrivacyError("workspace_home must be a non-root absolute path")
    return normalized


def _new_sanitizer(workspace_home: str | None) -> PIISanitizer:
    return PIISanitizer(workspace_home=_normalized_workspace_home(workspace_home))


def _pattern_descriptor(rule: tuple[str, PIICategory, re.Pattern[str], str]) -> dict[str, Any]:
    name, category, pattern, replacement = rule
    return {
        "name": name,
        "category": category.value,
        "pattern": pattern.pattern,
        "flags": pattern.flags,
        "replacement": replacement,
    }


def _ruleset_manifest_v1(sanitizer: PIISanitizer) -> dict[str, Any]:
    """Describe the frozen regex-oriented v1 policy."""

    sensitive_rules = (
        sanitizer.sensitive_registry.get_rules() if sanitizer.sensitive_registry is not None else []
    )
    workspace_pattern_hash = hashlib.sha256(
        sanitizer._workspace_home_pattern.pattern.encode("utf-8")
    ).hexdigest()
    return {
        "identity": RELEASE_PRIVACY_RULESET_IDENTITY_V1,
        "canonical_json": {
            "sort_keys": True,
            "separators": [",", ":"],
            "ensure_ascii": False,
            "allow_nan": False,
        },
        "scope": "all_json_string_values_and_object_keys",
        "key_collision_policy": "fail_closed",
        "residual_finding_policy": "fail_closed",
        "tokens": {
            category.value: DEFAULT_REDACTION_TOKENS[category]
            for category in sorted(DEFAULT_REDACTION_TOKENS, key=lambda item: item.value)
        },
        "builtin_rules": [_pattern_descriptor(rule) for rule in sanitizer._rules],
        "sensitive_phrase_rules": [_pattern_descriptor(rule) for rule in sensitive_rules],
        # The configured home affects the actual rule patterns.  Bind it without
        # placing the operator's path in the release receipt.
        "workspace_home_pattern_sha256": workspace_pattern_hash,
        "workspace_home_replacement_root": "/workspace",
    }


def _ruleset_manifest_v2(sanitizer: PIISanitizer) -> dict[str, Any]:
    """Describe every v2 detector, threshold, coverage, and allowance."""

    manifest = _ruleset_manifest_v1(sanitizer)
    manifest["identity"] = RELEASE_PRIVACY_RULESET_IDENTITY
    manifest["entropy_fallback"] = {
        "algorithm": "shannon-entropy-bits-per-character-v1",
        "candidate_pattern": _ENTROPY_CANDIDATE_RE.pattern,
        "candidate_pattern_flags": _ENTROPY_CANDIDATE_RE.flags,
        "minimum_length": _ENTROPY_MIN_LENGTH,
        "minimum_bits_per_character": _ENTROPY_MIN_BITS_PER_CHARACTER,
        "minimum_character_classes": _ENTROPY_MIN_CHARACTER_CLASSES,
        "character_classes": [
            "ascii_lowercase",
            "ascii_uppercase",
            "ascii_decimal_digit",
            "ascii_non_alphanumeric",
        ],
        "hex64_candidate_pattern": _HEX64_CANDIDATE_RE.pattern,
        "hex64_minimum_bits_per_character": _HEX64_MIN_BITS_PER_CHARACTER,
        "category": PIICategory.SECRET.value,
        "replacement": DEFAULT_REDACTION_TOKENS[PIICategory.SECRET],
        "structured_sha256_allowance": {
            "value_pattern": _SHA256_RE.pattern,
            "exact_field_names": sorted(_STRUCTURED_SHA256_EXACT_FIELDS),
            "prefixed_fields": dict(sorted(_STRUCTURED_PREFIXED_SHA256_FIELDS.items())),
            "scope": "entire_string_value_only",
            "field_context": "nearest_object_field_including_array_descendants",
        },
    }
    manifest["coverage"] = {
        "scope": _COVERAGE_SCOPE,
        "detectors": list(_COVERAGE_DETECTORS),
        "phase_fields": sorted(_COVERAGE_PHASE_FIELDS),
        "completion_policy": "expected_and_scanned_counts_must_match",
    }
    return manifest


def _ruleset_sha256_v1_for(sanitizer: PIISanitizer) -> str:
    return _sha256_json(_ruleset_manifest_v1(sanitizer))


def _ruleset_sha256_for(sanitizer: PIISanitizer) -> str:
    return _sha256_json(_ruleset_manifest_v2(sanitizer))


def _expected_v1_ruleset_sha256(workspace_home: str | None) -> str:
    if workspace_home is None:
        return _PORTABLE_V1_RULESET_SHA256
    return _ruleset_sha256_v1_for(_new_sanitizer(workspace_home))


def release_ruleset_sha256(*, workspace_home: str | None = None) -> str:
    """Hash the complete active detector/redaction rule manifest."""

    return _ruleset_sha256_for(_new_sanitizer(workspace_home))


def _empty_text_metrics() -> dict[str, int]:
    return {"object_key_strings": 0, "string_values": 0, "utf8_bytes": 0}


def _record_text_visit(metrics: dict[str, int], text: str, *, is_key: bool) -> None:
    metrics["object_key_strings" if is_key else "string_values"] += 1
    metrics["utf8_bytes"] += len(text.encode("utf-8"))


def _expected_text_metrics(value: Any) -> dict[str, int]:
    """Count scannable text independently from the detector traversal."""

    metrics = _empty_text_metrics()

    def _walk(item: Any) -> None:
        if type(item) is str:
            _record_text_visit(metrics, item, is_key=False)
        elif type(item) is dict:
            for key, child in item.items():
                _record_text_visit(metrics, key, is_key=True)
                _walk(child)
        elif type(item) is list:
            for child in item:
                _walk(child)

    _walk(value)
    return metrics


def _character_class_count(value: str) -> int:
    return sum(
        (
            any(character.islower() for character in value),
            any(character.isupper() for character in value),
            any(character.isdigit() for character in value),
            any(not character.isalnum() for character in value),
        )
    )


def _shannon_entropy(value: str) -> float:
    counts = Counter(value)
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def _is_allowed_structured_sha256(value: str, field_name: str | None) -> bool:
    """Allow only schema-shaped digest fields, never arbitrary 64-hex text."""

    if field_name is None:
        return False
    if _SHA256_RE.fullmatch(value) is not None:
        return field_name in _STRUCTURED_SHA256_EXACT_FIELDS
    prefix = _STRUCTURED_PREFIXED_SHA256_FIELDS.get(field_name)
    return bool(
        prefix is not None
        and value.startswith(prefix)
        and _SHA256_RE.fullmatch(value[len(prefix) :]) is not None
    )


def _entropy_findings(
    text: str,
    *,
    path: str,
    field_name: str | None,
) -> list[PIIFinding]:
    if _is_allowed_structured_sha256(text, field_name):
        return []

    findings: list[PIIFinding] = []
    for match in _ENTROPY_CANDIDATE_RE.finditer(text):
        candidate = match.group(0)
        entropy = _shannon_entropy(candidate)
        is_hex64 = _HEX64_CANDIDATE_RE.fullmatch(candidate) is not None
        if (
            len(candidate) >= _ENTROPY_MIN_LENGTH
            and (
                (is_hex64 and entropy >= _HEX64_MIN_BITS_PER_CHARACTER)
                or (
                    not is_hex64
                    and _character_class_count(candidate) >= _ENTROPY_MIN_CHARACTER_CLASSES
                    and entropy >= _ENTROPY_MIN_BITS_PER_CHARACTER
                )
            )
        ):
            findings.append(
                PIIFinding(
                    rule="high_entropy_token",
                    category=PIICategory.SECRET,
                    matched_text=candidate,
                    start=match.start(),
                    end=match.end(),
                    replacement=DEFAULT_REDACTION_TOKENS[PIICategory.SECRET],
                    confidence=0.75,
                    path=path,
                )
            )
    return findings


def _redact_release_text(
    text: str,
    *,
    path: str,
    field_name: str | None,
    sanitizer: PIISanitizer,
) -> tuple[str, list[PIIFinding]]:
    patterned, pattern_findings = sanitizer.redact_text(text, path=path)
    entropy_findings = _entropy_findings(
        patterned,
        path=path,
        field_name=field_name,
    )
    if not entropy_findings:
        return patterned, pattern_findings

    characters = list(patterned)
    for finding in reversed(entropy_findings):
        characters[finding.start : finding.end] = list(finding.replacement)
    return "".join(characters), [*pattern_findings, *entropy_findings]


def _redact_release_tree(
    value: Any,
    sanitizer: PIISanitizer,
) -> tuple[Any, list[PIIFinding], dict[str, int]]:
    findings: list[PIIFinding] = []
    scanned = _empty_text_metrics()

    def _walk(item: Any, *, path: str, field_name: str | None) -> Any:
        if type(item) is str:
            _record_text_visit(scanned, item, is_key=False)
            cleaned, item_findings = _redact_release_text(
                item,
                path=path,
                field_name=field_name,
                sanitizer=sanitizer,
            )
            findings.extend(item_findings)
            return cleaned
        if type(item) is dict:
            cleaned: dict[str, Any] = {}
            for index, (key, child) in enumerate(item.items()):
                _record_text_visit(scanned, key, is_key=True)
                cleaned_key, key_findings = _redact_release_text(
                    key,
                    path=f"{path}.key[{index}]",
                    field_name=None,
                    sanitizer=sanitizer,
                )
                findings.extend(key_findings)
                if cleaned_key in cleaned:
                    raise ReleasePrivacyError(
                        f"privacy redaction would collide object keys at {path}"
                    )
                cleaned[cleaned_key] = _walk(
                    child,
                    path=f"{path}.value[{index}]",
                    field_name=key,
                )
            return cleaned
        if type(item) is list:
            return [
                _walk(child, path=f"{path}[{index}]", field_name=field_name)
                for index, child in enumerate(item)
            ]
        return item

    return _walk(value, path="$", field_name=None), findings, scanned


def _scan_release_tree(
    value: Any,
    sanitizer: PIISanitizer,
) -> tuple[list[PIIFinding], dict[str, int]]:
    findings: list[PIIFinding] = []
    scanned = _empty_text_metrics()

    def _scan_text(text: str, *, path: str, field_name: str | None, is_key: bool) -> None:
        _record_text_visit(scanned, text, is_key=is_key)
        findings.extend(sanitizer.scan_text(text, path=path))
        findings.extend(
            _entropy_findings(
                text,
                path=path,
                field_name=field_name,
            )
        )

    def _walk(item: Any, *, path: str, field_name: str | None) -> None:
        if type(item) is str:
            _scan_text(item, path=path, field_name=field_name, is_key=False)
        elif type(item) is dict:
            for index, (key, child) in enumerate(item.items()):
                _scan_text(
                    key,
                    path=f"{path}.key[{index}]",
                    field_name=None,
                    is_key=True,
                )
                _walk(child, path=f"{path}.value[{index}]", field_name=key)
        elif type(item) is list:
            for index, child in enumerate(item):
                _walk(child, path=f"{path}[{index}]", field_name=field_name)

    _walk(value, path="$", field_name=None)
    return findings, scanned


def _coverage_phase(
    expected: dict[str, int],
    scanned: dict[str, int],
) -> dict[str, Any]:
    return {
        "object_key_strings_total": expected["object_key_strings"],
        "object_key_strings_scanned": scanned["object_key_strings"],
        "string_values_total": expected["string_values"],
        "string_values_scanned": scanned["string_values"],
        "utf8_bytes_total": expected["utf8_bytes"],
        "utf8_bytes_scanned": scanned["utf8_bytes"],
        "complete": expected == scanned,
    }


def _category_counts(findings: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        category = finding.category.value
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _receipt_sha256(receipt: dict[str, Any]) -> str:
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    return _sha256_json(body)


def sanitize_release_row(
    row: dict[str, Any],
    *,
    record_id: str,
    workspace_home: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sanitize a complete model-facing row and return a privacy-safe receipt.

    Any malformed JSON, redacted-key collision, or residual finding raises
    :class:`ReleasePrivacyError`; there is no permissive/HOLD return path at
    this final boundary.
    """

    if type(row) is not dict:
        raise ReleasePrivacyError("release row must be a JSON object")
    if type(record_id) is not str or not record_id:
        raise ReleasePrivacyError("record_id must be a non-empty string")
    _validate_json_tree(row)
    input_hash = _sha256_json(row)

    sanitizer = _new_sanitizer(workspace_home)
    expected_input_coverage = _expected_text_metrics(row)
    output_row, findings, scanned_input_coverage = _redact_release_tree(row, sanitizer)
    input_coverage = _coverage_phase(expected_input_coverage, scanned_input_coverage)
    if input_coverage["complete"] is not True:
        raise ReleasePrivacyError("privacy input traversal coverage is incomplete")
    if type(output_row) is not dict:  # defensive: the public input contract is a dict
        raise ReleasePrivacyError("sanitizer did not return a JSON object")
    _validate_json_tree(output_row)
    output_hash = _sha256_json(output_row)

    expected_output_coverage = _expected_text_metrics(output_row)
    residual_findings, scanned_output_coverage = _scan_release_tree(output_row, sanitizer)
    residual_output_coverage = _coverage_phase(
        expected_output_coverage,
        scanned_output_coverage,
    )
    if residual_output_coverage["complete"] is not True:
        raise ReleasePrivacyError("privacy residual traversal coverage is incomplete")
    if residual_findings:
        categories = ",".join(sorted(_category_counts(residual_findings)))
        raise ReleasePrivacyError(
            "privacy sanitizer left residual findings "
            f"({len(residual_findings)}; categories={categories})"
        )

    findings_by_category = _category_counts(findings)
    output_changed = input_hash != output_hash
    if bool(findings) != output_changed:
        raise ReleasePrivacyError("sanitizer findings and canonical row change disagree")

    receipt: dict[str, Any] = {
        "schema_id": RELEASE_PRIVACY_RECEIPT_SCHEMA,
        "record_id_sha256": _sha256_text(record_id),
        "ruleset_identity": RELEASE_PRIVACY_RULESET_IDENTITY,
        "ruleset_sha256": _ruleset_sha256_for(sanitizer),
        "input_row_sha256": input_hash,
        "output_row_sha256": output_hash,
        "findings_count": len(findings),
        "findings_by_category": findings_by_category,
        "output_changed": output_changed,
        "residual_findings_count": 0,
        "residual_findings_by_category": {},
        "status": _PASS_REDACTION if output_changed else _PASS_SCAN,
        "coverage": {
            "scope": _COVERAGE_SCOPE,
            "detectors": list(_COVERAGE_DETECTORS),
            "input": input_coverage,
            "residual_output": residual_output_coverage,
        },
    }
    receipt["receipt_sha256"] = _receipt_sha256(receipt)
    return output_row, receipt


def _is_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _valid_category_counts(value: Any) -> bool:
    if type(value) is not dict:
        return False
    known = {category.value for category in PIICategory}
    return all(
        type(key) is str
        and key in known
        and type(count) is int
        and count >= 0
        for key, count in value.items()
    )


def _valid_coverage_phase(value: Any) -> bool:
    if type(value) is not dict or frozenset(value) != _COVERAGE_PHASE_FIELDS:
        return False
    count_fields = _COVERAGE_PHASE_FIELDS - {"complete"}
    if not all(type(value.get(field)) is int and value[field] >= 0 for field in count_fields):
        return False
    if type(value.get("complete")) is not bool:
        return False
    counts_complete = (
        value["object_key_strings_total"] == value["object_key_strings_scanned"]
        and value["string_values_total"] == value["string_values_scanned"]
        and value["utf8_bytes_total"] == value["utf8_bytes_scanned"]
    )
    return value["complete"] is counts_complete


def _coverage_errors(
    coverage: Any,
    output_row: dict[str, Any],
    sanitizer: PIISanitizer,
) -> tuple[list[str], list[PIIFinding]]:
    if type(coverage) is not dict:
        return ["receipt coverage is malformed"], []
    expected_fields = {"scope", "detectors", "input", "residual_output"}
    if set(coverage) != expected_fields:
        return ["receipt coverage fields differ"], []

    errors: list[str] = []
    if coverage.get("scope") != _COVERAGE_SCOPE:
        errors.append("receipt coverage scope mismatch")
    if coverage.get("detectors") != list(_COVERAGE_DETECTORS):
        errors.append("receipt coverage detectors mismatch")

    input_phase = coverage.get("input")
    residual_phase = coverage.get("residual_output")
    if not _valid_coverage_phase(input_phase):
        errors.append("receipt input coverage is malformed")
    elif input_phase.get("complete") is not True:
        errors.append("receipt input coverage is incomplete")
    if not _valid_coverage_phase(residual_phase):
        errors.append("receipt residual coverage is malformed")
    elif residual_phase.get("complete") is not True:
        errors.append("receipt residual coverage is incomplete")

    actual_expected = _expected_text_metrics(output_row)
    actual_residual, actual_scanned = _scan_release_tree(output_row, sanitizer)
    actual_phase = _coverage_phase(actual_expected, actual_scanned)
    if type(residual_phase) is dict and residual_phase != actual_phase:
        errors.append("receipt residual coverage differs from output traversal")

    if type(input_phase) is dict and type(residual_phase) is dict:
        for total_field in ("object_key_strings_total", "string_values_total"):
            if input_phase.get(total_field) != residual_phase.get(total_field):
                errors.append("receipt input/output structural coverage differs")
                break
    return errors, actual_residual


def verify_release_privacy_receipt_integrity(
    receipt: dict[str, Any],
    output_row: dict[str, Any],
    *,
    workspace_home: str | None = None,
    record_id: str | None = None,
) -> ReleasePrivacyVerification:
    """Verify a known receipt policy while preserving legacy-only status."""

    if type(receipt) is not dict:
        return ReleasePrivacyVerification(_INVALID, ("receipt must be a JSON object",))

    errors: list[str] = []
    try:
        _validate_json_tree(receipt)
        if type(output_row) is not dict:
            raise ReleasePrivacyError("release output row must be a JSON object")
        _validate_json_tree(output_row)
        actual_output_hash = _sha256_json(output_row)
        sanitizer = _new_sanitizer(workspace_home)
    except ReleasePrivacyError as exc:
        return ReleasePrivacyVerification(_INVALID, (str(exc),))

    policy_tuple = (receipt.get("schema_id"), receipt.get("ruleset_identity"))
    if policy_tuple == (RELEASE_PRIVACY_RECEIPT_SCHEMA, RELEASE_PRIVACY_RULESET_IDENTITY):
        expected_fields = _RECEIPT_FIELDS_V2
        expected_ruleset_hash: str | None = _ruleset_sha256_for(sanitizer)
        verified_status = _CURRENT_POLICY_VERIFIED
    elif policy_tuple == (
        RELEASE_PRIVACY_RECEIPT_SCHEMA_V1,
        RELEASE_PRIVACY_RULESET_IDENTITY_V1,
    ):
        expected_fields = _RECEIPT_FIELDS_V1
        expected_ruleset_hash = _expected_v1_ruleset_sha256(workspace_home)
        verified_status = _HISTORICAL_INTEGRITY_ONLY
    else:
        expected_fields = frozenset(receipt)
        expected_ruleset_hash = None
        verified_status = _INVALID
        errors.append("receipt privacy policy tuple is unknown")

    fields = frozenset(receipt)
    if fields != expected_fields:
        missing = sorted(expected_fields - fields)
        unknown = sorted(fields - expected_fields)
        if missing:
            errors.append(f"receipt is missing fields: {','.join(missing)}")
        if unknown:
            # Unknown field names are untrusted receipt material and may
            # themselves be sensitive; report only their count.
            errors.append(f"receipt has {len(unknown)} unknown field(s)")

    if expected_ruleset_hash is not None and receipt.get("ruleset_sha256") != expected_ruleset_hash:
        errors.append("receipt ruleset hash mismatch")

    for field in (
        "record_id_sha256",
        "ruleset_sha256",
        "input_row_sha256",
        "output_row_sha256",
        "receipt_sha256",
    ):
        if not _is_sha256(receipt.get(field)):
            errors.append(f"receipt {field} is not a SHA-256 digest")

    if receipt.get("output_row_sha256") != actual_output_hash:
        errors.append("receipt output row hash mismatch")
    if record_id is not None:
        if type(record_id) is not str or not record_id:
            errors.append("record_id must be a non-empty string")
        else:
            try:
                expected_record_id_hash = _sha256_text(record_id)
            except ReleasePrivacyError as exc:
                errors.append(str(exc))
            else:
                if receipt.get("record_id_sha256") != expected_record_id_hash:
                    errors.append("receipt record_id hash mismatch")

    findings_count = receipt.get("findings_count")
    counts = receipt.get("findings_by_category")
    if type(findings_count) is not int or findings_count < 0:
        errors.append("receipt findings_count must be a non-negative integer")
    if not _valid_category_counts(counts):
        errors.append("receipt findings_by_category is malformed")
    elif type(findings_count) is int and sum(counts.values()) != findings_count:
        errors.append("receipt finding category counts do not sum to findings_count")

    output_changed = receipt.get("output_changed")
    if type(output_changed) is not bool:
        errors.append("receipt output_changed must be boolean")
    hashes_changed = receipt.get("input_row_sha256") != receipt.get("output_row_sha256")
    if type(output_changed) is bool and output_changed != hashes_changed:
        errors.append("receipt output_changed disagrees with row hashes")

    residual_count = receipt.get("residual_findings_count")
    residual_counts = receipt.get("residual_findings_by_category")
    if residual_count != 0 or residual_counts != {}:
        errors.append("receipt claims residual privacy findings")

    status = receipt.get("status")
    expected_status = _PASS_REDACTION if output_changed is True else _PASS_SCAN
    if status != expected_status:
        errors.append("receipt status disagrees with output_changed")
    if type(findings_count) is int and type(output_changed) is bool:
        if (findings_count > 0) != output_changed:
            errors.append("receipt findings_count disagrees with output_changed")

    if verified_status == _CURRENT_POLICY_VERIFIED:
        coverage_defects, actual_residual = _coverage_errors(
            receipt.get("coverage"),
            output_row,
            sanitizer,
        )
        errors.extend(coverage_defects)
        if output_changed is False and type(receipt.get("coverage")) is dict:
            coverage = receipt["coverage"]
            if coverage.get("input") != coverage.get("residual_output"):
                errors.append("unchanged row input/output coverage differs")
    elif verified_status == _HISTORICAL_INTEGRITY_ONLY:
        # V1 is checked only under its frozen regex-oriented semantics.  It is
        # never promoted into current admission by this integrity API.
        actual_residual = sanitizer.scan_object(output_row)
    else:
        actual_residual = []
    if actual_residual:
        errors.append("output row still contains privacy findings")

    claimed_receipt_hash = receipt.get("receipt_sha256")
    try:
        actual_receipt_hash = _receipt_sha256(receipt)
    except ReleasePrivacyError:
        errors.append("receipt is not canonical JSON")
    else:
        if not (
            type(claimed_receipt_hash) is str
            and hmac.compare_digest(claimed_receipt_hash, actual_receipt_hash)
        ):
            errors.append("receipt self-hash mismatch")

    if errors:
        return ReleasePrivacyVerification(_INVALID, tuple(errors))
    return ReleasePrivacyVerification(verified_status, ())


def verify_release_privacy_receipt(
    receipt: dict[str, Any],
    output_row: dict[str, Any],
    *,
    workspace_home: str | None = None,
    record_id: str | None = None,
) -> list[str]:
    """Require the current v2 policy; an empty list is the only admission pass."""

    verification = verify_release_privacy_receipt_integrity(
        receipt,
        output_row,
        workspace_home=workspace_home,
        record_id=record_id,
    )
    errors = list(verification.errors)
    if verification.status == _HISTORICAL_INTEGRITY_ONLY:
        errors.append(
            "historical v1 receipt is integrity-only and cannot satisfy current admission"
        )
    return errors
