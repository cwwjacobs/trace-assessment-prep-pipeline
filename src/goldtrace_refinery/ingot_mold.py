"""Deterministic, declarative domain-Ingot casting.

The mold consumes an already admitted three-tier Refinery Ingot plus one
domain record.  It does not call a model, execute mold-provided code, sign a
receipt, invoke Hallmark, or submit to Vault.  Its two outputs are byte-stable:

* ``ingot-core.json`` is a semantic core whose identity ignores upstream
  timestamped issuance wrappers while preserving their stable decision facts.
* ``cast-request.json`` binds the exact input files and remains explicitly
  unsigned and on HOLD for independent issuance, Hallmark, Vault recomputation,
  and human sale authorization.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .hashing import canonical_json, sha256_bytes, sha256_json
from .ingot_schema import validate_ingot
from .paths import find_contract

CORE_SCHEMA_FILENAME = "domain-ingot-core.v1.schema.json"
CAST_REQUEST_SCHEMA_FILENAME = "domain-ingot-cast-request.v1.schema.json"
MOLD_SCHEMA_FILENAME = "ingot-mold-module.v1.schema.json"
DEFAULT_MOLD_FILENAME = "sentinel-sanctuary.cyber.v1.json"

CORE_SCHEMA_ID = "goldtrace.refinery.domain_ingot_core.v1"
CAST_REQUEST_SCHEMA_ID = "goldtrace.refinery.domain_ingot_cast_request.v1"
THREE_TIER_PATH = "three_tier_labyrinth_foundry_refinery"
MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_SAFE_INTEGER = 9_007_199_254_740_991

CORE_ARTIFACT_NAME = "ingot-core.json"
CAST_REQUEST_ARTIFACT_NAME = "cast-request.json"
_SORTED_RECORD_LIST_PATHS = (
    ("finding_codes",),
    ("evidence_ref_sha256s",),
    ("decision_episode_sha256s",),
    ("gap_codes",),
    ("split_group_ids",),
    ("lineage", "compared_core_sha256s"),
    ("evaluation", "evaluation_case_sha256s"),
    ("evaluation", "evaluation_result_sha256s"),
)


class MoldError(ValueError):
    """A mold input, policy gate, or deterministic artifact is invalid."""


@dataclass(frozen=True)
class LoadedMold:
    """A schema-validated declarative mold and its exact identities."""

    path: Path
    document: dict[str, Any]
    file_sha256: str
    canonical_sha256: str
    record_schema: dict[str, Any]
    record_schema_canonical_sha256: str


def _reject_float(_value: str) -> Any:
    raise MoldError("floating-point JSON values are forbidden")


def _reject_constant(_value: str) -> Any:
    raise MoldError("non-finite JSON values are forbidden")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MoldError("duplicate JSON object key is forbidden")
        result[key] = value
    return result


def _assert_safe_json(value: Any) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise MoldError("JSON integer exceeds the portable exact range")
        return
    if isinstance(value, float):
        raise MoldError("floating-point JSON values are forbidden")
    if isinstance(value, list):
        for item in value:
            _assert_safe_json(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise MoldError("JSON object keys must be strings")
            _assert_safe_json(item)
        return
    raise MoldError("value is outside the deterministic JSON data model")


def _load_json_bytes(data: bytes, *, subject: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except MoldError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MoldError(f"{subject} is not one valid UTF-8 JSON document") from exc
    if not isinstance(value, dict):
        raise MoldError(f"{subject} must be a JSON object")
    _assert_safe_json(value)
    return value


def _stable_read(path: Path | str, *, subject: str) -> tuple[Path, bytes]:
    """Read one bounded regular file through one ``O_NOFOLLOW`` descriptor."""

    absolute = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise MoldError(f"{subject} must be a readable regular non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise MoldError(f"{subject} must be a regular file")
        if before.st_size > MAX_INPUT_BYTES:
            raise MoldError(f"{subject} exceeds the {MAX_INPUT_BYTES}-byte limit")
        chunks: list[bytes] = []
        remaining = MAX_INPUT_BYTES + 1
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    if len(data) > MAX_INPUT_BYTES:
        raise MoldError(f"{subject} exceeds the {MAX_INPUT_BYTES}-byte limit")
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise MoldError(f"{subject} changed while it was read")
    if len(data) != after.st_size:
        raise MoldError(f"{subject} could not be read completely")
    return absolute, data


def _schema_document(filename: str) -> dict[str, Any]:
    path, data = _stable_read(find_contract(filename), subject=f"contract {filename}")
    del path
    schema = _load_json_bytes(data, subject=f"contract {filename}")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise MoldError(f"contract {filename} is not a valid JSON Schema") from exc
    return schema


def _validate(instance: Any, schema: dict[str, Any], *, subject: str) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(instance),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    first = errors[0]
    location = "/".join(str(part) for part in first.absolute_path) or "<root>"
    keyword = str(first.validator or "schema")
    raise MoldError(
        f"{subject} violates its schema at {location} ({keyword}); "
        f"{len(errors)} violation(s) total"
    )


def load_mold(path: Path | str | None = None) -> LoadedMold:
    """Load a declarative mold and verify its separately bound record schema."""

    selected = find_contract(DEFAULT_MOLD_FILENAME) if path is None else Path(path)
    mold_path, mold_bytes = _stable_read(selected, subject="mold module")
    document = _load_json_bytes(mold_bytes, subject="mold module")
    _validate(document, _schema_document(MOLD_SCHEMA_FILENAME), subject="mold module")

    schema_filename = document["record_schema_filename"]
    if Path(schema_filename).name != schema_filename:
        raise MoldError("mold record schema filename must not contain a path")
    adjacent_schema = mold_path.parent / schema_filename
    schema_path = adjacent_schema if adjacent_schema.is_file() else find_contract(schema_filename)
    _, schema_bytes = _stable_read(schema_path, subject="mold record schema")
    record_schema = _load_json_bytes(schema_bytes, subject="mold record schema")
    try:
        Draft202012Validator.check_schema(record_schema)
    except SchemaError as exc:
        raise MoldError("mold record schema is not a valid JSON Schema") from exc
    schema_hash = sha256_json(record_schema)
    if schema_hash != document["record_schema_sha256"]:
        raise MoldError("mold record schema canonical hash does not match the module")
    if record_schema.get("$id") != document["record_schema_id"]:
        raise MoldError("mold record schema identity does not match the module")

    return LoadedMold(
        path=mold_path,
        document=document,
        file_sha256=sha256_bytes(mold_bytes),
        canonical_sha256=sha256_json(document),
        record_schema=record_schema,
        record_schema_canonical_sha256=schema_hash,
    )


def _require_sha256(value: Any, *, subject: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise MoldError(f"{subject} must be a lowercase SHA-256")
    return value


def _require_nonempty_string(value: Any, *, subject: str) -> str:
    if not isinstance(value, str) or not value:
        raise MoldError(f"{subject} must be a non-empty string")
    return value


def _normalized_source(ingot: dict[str, Any]) -> dict[str, Any]:
    """Project only stable, independently checked upstream decision facts."""

    if validate_ingot(ingot):
        raise MoldError("source Ingot failed the Refinery Ingot contract")
    if ingot.get("evaluation_path") != THREE_TIER_PATH:
        raise MoldError("source Ingot did not traverse Labyrinth, Foundry, and Refinery")
    if ingot.get("refinery_status") != "PASSED":
        raise MoldError("source Ingot is not Refinery PASSED")
    if ingot.get("mechanical_run_status") != "COMPLETED":
        raise MoldError("source Ingot run is not mechanically complete")
    if ingot.get("seal_status") not in {"SEALED", "SEALED_WITH_GAPS"}:
        raise MoldError("source Ingot does not bind a sealed Labyrinth bundle")

    quarantine = ingot.get("quarantine")
    if not isinstance(quarantine, dict) or quarantine.get("status") != "CLEAR":
        raise MoldError("source Ingot is not quarantine CLEAR")
    redaction = ingot.get("redaction")
    unresolved_high = (
        redaction.get("unresolved_high_severity")
        if isinstance(redaction, dict)
        else None
    )
    if (
        not isinstance(unresolved_high, int)
        or isinstance(unresolved_high, bool)
        or unresolved_high != 0
    ):
        raise MoldError("source Ingot has unresolved high-severity privacy findings")

    foundry = ingot.get("foundry_evaluation")
    if not isinstance(foundry, dict) or foundry.get("status") != "PASS":
        raise MoldError("source Ingot lacks an admissible Foundry PASS")
    if foundry.get("deterministic") is not True:
        raise MoldError("Foundry decision core is not marked deterministic")
    trust = foundry.get("trust")
    if not isinstance(trust, dict) or (
        trust.get("pack_identity_verified") is not True
        or trust.get("evaluation_consistency_verified") is not True
    ):
        raise MoldError("source Ingot lacks verified Foundry trust facts")

    source_gap_ids = ingot.get("evidence_gap_ids") or []
    foundry_gap_ids = foundry.get("evidence_gaps") or []
    if not isinstance(source_gap_ids, list) or not all(
        isinstance(item, str) and item for item in source_gap_ids
    ):
        raise MoldError("source Ingot evidence gaps are malformed")
    if not isinstance(foundry_gap_ids, list) or not all(
        isinstance(item, str) and item for item in foundry_gap_ids
    ):
        raise MoldError("Foundry evidence gaps are malformed")

    return {
        "source_ingot_id": _require_nonempty_string(
            ingot.get("ingot_id"), subject="source Ingot id"
        ),
        "source_bundle_sha256": _require_sha256(
            ingot.get("source_bundle_hash"), subject="source bundle hash"
        ),
        "source_manifest_sha256": _require_sha256(
            ingot.get("source_manifest_hash"), subject="source manifest hash"
        ),
        "normalized_event_stream_sha256": _require_sha256(
            ingot.get("normalized_event_stream_hash"),
            subject="normalized event stream hash",
        ),
        "scenario_id": _require_nonempty_string(
            ingot.get("scenario_id"), subject="scenario id"
        ),
        "scenario_version": _require_nonempty_string(
            ingot.get("scenario_version"), subject="scenario version"
        ),
        "mine_run_id": ingot.get("mine_run_id"),
        "source_seal_status": ingot["seal_status"],
        "evaluation_path": THREE_TIER_PATH,
        "foundry_status": "PASS",
        "evaluation_pack_id": _require_nonempty_string(
            foundry.get("evaluation_pack_id"), subject="evaluation pack id"
        ),
        "evaluation_pack_version": _require_nonempty_string(
            foundry.get("evaluation_pack_version"), subject="evaluation pack version"
        ),
        "evaluation_pack_sha256": _require_sha256(
            foundry.get("evaluation_pack_hash"), subject="evaluation pack hash"
        ),
        "foundry_code_sha256": _require_sha256(
            foundry.get("foundry_code_sha256"), subject="Foundry code hash"
        ),
        "foundry_decision_deterministic": True,
        "foundry_issuer_id": _require_nonempty_string(
            trust.get("issuer_id"), subject="Foundry issuer id"
        ),
        "foundry_key_id": _require_nonempty_string(
            trust.get("key_id"), subject="Foundry key id"
        ),
        "foundry_authentication": _require_nonempty_string(
            trust.get("channel"), subject="Foundry authentication method"
        ),
        "foundry_evidence_gap_ids": sorted(set(source_gap_ids + foundry_gap_ids)),
        "refinery_status": "PASSED",
    }


def _normalize_record(record: dict[str, Any], mold: LoadedMold) -> dict[str, Any]:
    _validate(record, mold.record_schema, subject="domain record")
    normalized = copy.deepcopy(record)
    for path in _SORTED_RECORD_LIST_PATHS:
        cursor: Any = normalized
        for part in path[:-1]:
            cursor = cursor[part]
        cursor[path[-1]] = sorted(cursor[path[-1]])

    policy = mold.document
    if normalized["schema_id"] != policy["record_schema_id"]:
        raise MoldError("domain record schema identity is outside the mold")
    if normalized["emission_type"] not in policy["allowed_emission_types"]:
        raise MoldError("domain record emission type is outside the mold")
    if normalized["case_type"] not in policy["allowed_case_types"]:
        raise MoldError("domain record case type is outside the mold")
    if normalized["replay_mode"] not in policy["replay_modes"]:
        raise MoldError("domain record replay mode is outside the mold")

    execution = normalized["execution_witness"]
    if execution["profile"] not in policy["allowed_execution_profiles"]:
        raise MoldError("Cinderfield execution profile is outside the mold")
    if execution["assurance_class"] not in policy["allowed_assurance_classes"]:
        raise MoldError("Cinderfield assurance class is outside the mold")

    model = normalized["model_witness"]
    if model["api_family"] not in policy["allowed_provider_api_families"]:
        raise MoldError("provider API family is outside the mold")
    if model["store_requested"] != policy["allow_provider_application_state"]:
        raise MoldError("provider application-state request is outside the mold")
    if model["output_reproducibility"] != policy["required_output_reproducibility"]:
        raise MoldError("model output reproducibility claim is outside the mold")

    if len(normalized["evidence_ref_sha256s"]) < policy["minimum_evidence_refs"]:
        raise MoldError("domain record has too few evidence references for the mold")
    if (
        len(normalized["decision_episode_sha256s"])
        < policy["minimum_decision_episodes"]
    ):
        raise MoldError("domain record has too few decision episodes for the mold")

    split_groups = normalized["split_group_ids"]
    prefixes = policy["required_split_group_prefixes"]
    for group in split_groups:
        if not any(group.startswith(prefix) and len(group) > len(prefix) for prefix in prefixes):
            raise MoldError("domain record contains an unrecognized split-group namespace")
    missing = [
        prefix
        for prefix in prefixes
        if not any(group.startswith(prefix) and len(group) > len(prefix) for group in split_groups)
    ]
    if missing:
        raise MoldError("domain record is missing a required split-group namespace")
    return normalized


def _core_hash_body(core: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in core.items()
        if key not in {"core_id", "core_sha256"}
    }


def build_domain_ingot_core(
    source_ingot: dict[str, Any],
    record: dict[str, Any],
    *,
    mold: LoadedMold | None = None,
) -> dict[str, Any]:
    """Build and self-verify one deterministic, always-HOLD domain core."""

    selected = mold or load_mold()
    source = _normalized_source(source_ingot)
    normalized_record = _normalize_record(record, selected)
    body = {
        "schema_id": CORE_SCHEMA_ID,
        "canonicalization": "goldtrace.canonical-json.v1",
        "mold": {
            "mold_id": selected.document["mold_id"],
            "mold_version": selected.document["mold_version"],
            "mold_sha256": selected.canonical_sha256,
            "domain": selected.document["domain"],
            "record_schema_id": selected.document["record_schema_id"],
            "record_schema_sha256": selected.record_schema_canonical_sha256,
        },
        "source": source,
        "record": normalized_record,
        "claims": {
            "execution_scope": "OBSERVED_RUN_ONLY",
            "model_output_determinism": "NOT_CLAIMED",
            "provider_disclosure_laundered": False,
            "automatic_promotion": False,
            "release_posture": "HOLD",
        },
    }
    digest = sha256_json(body)
    core = {
        **body,
        "core_id": f"GTIC-{digest[:24]}",
        "core_sha256": digest,
    }
    verify_domain_ingot_core(core, mold=selected)
    return core


def verify_domain_ingot_core(
    core: dict[str, Any],
    *,
    mold: LoadedMold | None = None,
) -> None:
    """Recompute all deterministic identities and the separately bound record."""

    selected = mold or load_mold()
    _assert_safe_json(core)
    _validate(core, _schema_document(CORE_SCHEMA_FILENAME), subject="domain Ingot core")
    _validate(core["record"], selected.record_schema, subject="domain Ingot record")
    mold_binding = core["mold"]
    expected_mold = {
        "mold_id": selected.document["mold_id"],
        "mold_version": selected.document["mold_version"],
        "mold_sha256": selected.canonical_sha256,
        "domain": selected.document["domain"],
        "record_schema_id": selected.document["record_schema_id"],
        "record_schema_sha256": selected.record_schema_canonical_sha256,
    }
    if mold_binding != expected_mold:
        raise MoldError("domain Ingot core does not bind the selected mold exactly")
    if _normalize_record(core["record"], selected) != core["record"]:
        raise MoldError("domain Ingot record is not in its normalized set ordering")
    digest = sha256_json(_core_hash_body(core))
    if core["core_sha256"] != digest or core["core_id"] != f"GTIC-{digest[:24]}":
        raise MoldError("domain Ingot core identity does not match its canonical body")


def _required_gates(core: dict[str, Any]) -> list[str]:
    record = core["record"]
    execution = record["execution_witness"]
    model = record["model_witness"]
    evaluation = record["evaluation"]
    governance = record["governance"]
    gates = {
        "AUTHENTICATED_ISSUANCE",
        "AUTHENTICATED_HALLMARK",
        "VAULT_OWNED_RECOMPUTATION",
        "HUMAN_SALE_RELEASE",
    }
    if core["source"]["foundry_authentication"] != "ed25519":
        gates.add("ASYMMETRIC_FOUNDRY_ATTESTATION")
    if execution["assurance_class"] != "SUPPORTED_HOST_VERIFIED":
        gates.add("SUPPORTED_HOST_CINDERFIELD_PROOF")
    if (
        core["source"]["source_seal_status"] == "SEALED_WITH_GAPS"
        or core["source"]["foundry_evidence_gap_ids"]
        or record["gap_codes"]
    ):
        gates.add("EVIDENCE_GAPS_RESOLVED")
    if governance["commercial_rights_status"] != "VERIFIED":
        gates.add("COMMERCIAL_RIGHTS_VERIFICATION")
    if evaluation["contamination_status"] != "CLEAR":
        gates.add("CONTAMINATION_CLEARANCE")
    if execution["evidence_recoverability"] != "DECRYPTABLE_VERIFIED":
        gates.add("EVIDENCE_DECRYPTABILITY_PROOF")
    if execution["observer_authentication"] != "ASYMMETRIC_VERIFIED":
        gates.add("ASYMMETRIC_EXECUTION_ATTESTATION")
    if execution["cleanup_status"] != "VERIFIED":
        gates.add("VERIFIED_CLEANUP")
    if execution["credential_channel_closed"] is not True:
        gates.add("CREDENTIAL_CHANNEL_CLOSURE")
    if execution["gate_decision"] in {"RESCOPE_REQUIRED", "EVALUATION_INCOMPLETE"}:
        gates.add("EXECUTION_SCOPE_RESOLUTION")
    if model["model_snapshot_pinned"] is not True:
        gates.add("PINNED_MODEL_SNAPSHOT")
    return sorted(gates)


def _request_hash_body(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in request.items()
        if key not in {"request_id", "request_sha256"}
    }


def build_cast_request(
    core: dict[str, Any],
    source_ingot: dict[str, Any],
    normalized_record: dict[str, Any],
    *,
    mold: LoadedMold,
    source_ingot_file_sha256: str,
    record_file_sha256: str,
) -> dict[str, Any]:
    """Bind exact input artifacts without granting issuance or release authority."""

    verify_domain_ingot_core(core, mold=mold)
    foundry = source_ingot["foundry_evaluation"]
    body = {
        "schema_id": CAST_REQUEST_SCHEMA_ID,
        "core_id": core["core_id"],
        "core_sha256": core["core_sha256"],
        "mold_id": mold.document["mold_id"],
        "mold_version": mold.document["mold_version"],
        "artifacts": {
            "source_ingot_file_sha256": _require_sha256(
                source_ingot_file_sha256, subject="source Ingot file hash"
            ),
            "source_refinery_receipt_sha256": _require_sha256(
                source_ingot.get("refinery_receipt_hash"),
                subject="source Refinery receipt hash",
            ),
            "foundry_receipt_sha256": _require_sha256(
                foundry.get("receipt_hash"), subject="Foundry receipt hash"
            ),
            "record_file_sha256": _require_sha256(
                record_file_sha256, subject="record file hash"
            ),
            "normalized_record_sha256": sha256_json(normalized_record),
            "mold_file_sha256": mold.file_sha256,
            "mold_canonical_sha256": mold.canonical_sha256,
            "record_schema_canonical_sha256": mold.record_schema_canonical_sha256,
        },
        "issuance": {
            "authentication_status": "UNSIGNED",
            "hallmark_status": "NOT_RUN",
            "vault_status": "NOT_SUBMITTED",
            "sale_eligibility": "HOLD",
            "required_gates": _required_gates(core),
        },
    }
    digest = sha256_json(body)
    request = {
        **body,
        "request_id": f"GTCR-{digest[:24]}",
        "request_sha256": digest,
    }
    verify_cast_request(request, core=core, mold=mold)
    return request


def verify_cast_request(
    request: dict[str, Any],
    *,
    core: dict[str, Any],
    mold: LoadedMold,
) -> None:
    _assert_safe_json(request)
    _validate(
        request,
        _schema_document(CAST_REQUEST_SCHEMA_FILENAME),
        subject="domain Ingot cast request",
    )
    if request["core_id"] != core["core_id"] or request["core_sha256"] != core["core_sha256"]:
        raise MoldError("cast request does not bind the supplied domain Ingot core")
    if (
        request["mold_id"] != mold.document["mold_id"]
        or request["mold_version"] != mold.document["mold_version"]
    ):
        raise MoldError("cast request does not bind the supplied mold")
    artifacts = request["artifacts"]
    if (
        artifacts["mold_file_sha256"] != mold.file_sha256
        or artifacts["mold_canonical_sha256"] != mold.canonical_sha256
        or artifacts["record_schema_canonical_sha256"]
        != mold.record_schema_canonical_sha256
    ):
        raise MoldError("cast request mold artifact identities do not match")
    if artifacts["normalized_record_sha256"] != sha256_json(core["record"]):
        raise MoldError("cast request normalized record identity does not match the core")
    if request["issuance"]["required_gates"] != _required_gates(core):
        raise MoldError("cast request gate set does not match the core")
    digest = sha256_json(_request_hash_body(request))
    if request["request_sha256"] != digest or request["request_id"] != f"GTCR-{digest[:24]}":
        raise MoldError("cast request identity does not match its canonical body")


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> bool:
    """Use Linux ``RENAME_NOREPLACE``; return false when unsupported."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:  # pragma: no cover - current runtime is Linux/glibc
        return False
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(destination)
    unsupported = {
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        getattr(errno, "ENOTSUP", errno.EINVAL),
    }
    if error in unsupported:
        return False
    raise MoldError("atomic cast installation failed") from OSError(
        error, os.strerror(error)
    )


def _existing_output_matches(output: Path, payloads: Mapping[str, bytes]) -> bool:
    try:
        metadata = output.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise MoldError("cast output must be a regular non-symlink directory")
    entries = {entry.name for entry in os.scandir(output)}
    if entries != set(payloads):
        raise MoldError("existing cast output inventory differs")
    for name, expected in payloads.items():
        _, actual = _stable_read(output / name, subject=f"existing cast output {name}")
        if actual != expected:
            raise MoldError("existing cast output bytes differ")
    return True


def _install_output(output: Path, payloads: Mapping[str, bytes]) -> None:
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_metadata = parent.lstat()
    if not stat.S_ISDIR(parent_metadata.st_mode) or stat.S_ISLNK(parent_metadata.st_mode):
        raise MoldError("cast output parent must be a regular non-symlink directory")
    if _existing_output_matches(output, payloads):
        return

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        for name in sorted(payloads):
            _write_private_file(temporary / name, payloads[name])
        _fsync_directory(temporary)
        try:
            installed = _rename_directory_noreplace(temporary, output)
        except FileExistsError:
            if _existing_output_matches(output, payloads):
                return
            raise MoldError("cast output appeared concurrently with different bytes")
        if installed:
            _fsync_directory(parent)
            return

        try:
            output.mkdir(mode=0o700)
        except FileExistsError:
            if _existing_output_matches(output, payloads):
                return
            raise MoldError("cast output appeared concurrently with different bytes")
        marker = output / ".INSTALLING"
        _write_private_file(marker, b"domain Ingot cast installation incomplete\n")
        for name in sorted(payloads):
            os.rename(temporary / name, output / name)
        temporary.rmdir()
        _fsync_directory(output)
        marker.unlink()
        _fsync_directory(output)
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def cast_files(
    source_ingot_path: Path | str,
    record_path: Path | str,
    output_path: Path | str,
    *,
    mold_path: Path | str | None = None,
) -> dict[str, str]:
    """Cast, verify, and idempotently install two deterministic HOLD artifacts."""

    _, source_bytes = _stable_read(source_ingot_path, subject="source Ingot")
    _, record_bytes = _stable_read(record_path, subject="domain record")
    source_ingot = _load_json_bytes(source_bytes, subject="source Ingot")
    record = _load_json_bytes(record_bytes, subject="domain record")
    mold = load_mold(mold_path)
    core = build_domain_ingot_core(source_ingot, record, mold=mold)
    normalized_record = core["record"]
    request = build_cast_request(
        core,
        source_ingot,
        normalized_record,
        mold=mold,
        source_ingot_file_sha256=sha256_bytes(source_bytes),
        record_file_sha256=sha256_bytes(record_bytes),
    )
    core_bytes = canonical_json(core) + b"\n"
    request_bytes = canonical_json(request) + b"\n"
    output = Path(os.path.abspath(os.fspath(Path(output_path).expanduser())))
    _install_output(
        output,
        {
            CORE_ARTIFACT_NAME: core_bytes,
            CAST_REQUEST_ARTIFACT_NAME: request_bytes,
        },
    )
    return {
        "core_id": core["core_id"],
        "core_sha256": core["core_sha256"],
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "release_posture": "HOLD",
        "core_path": str(output / CORE_ARTIFACT_NAME),
        "cast_request_path": str(output / CAST_REQUEST_ARTIFACT_NAME),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cast a deterministic, unsigned, always-HOLD domain Ingot core"
    )
    parser.add_argument("--ingot", required=True, type=Path, help="three-tier Refinery ingot.json")
    parser.add_argument("--record", required=True, type=Path, help="domain record JSON")
    parser.add_argument(
        "--out", required=True, type=Path, help="new or byte-identical output directory"
    )
    parser.add_argument("--mold", type=Path, help="optional declarative mold module")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = cast_files(args.ingot, args.record, args.out, mold_path=args.mold)
    except MoldError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
