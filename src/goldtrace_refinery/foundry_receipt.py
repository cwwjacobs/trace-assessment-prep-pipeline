"""
Verification of Tier-2 Foundry evaluation receipts.

Refinery will not cast a three-tier ingot without a receipt proving that Foundry
verified and evaluated the *same* bundle. Every check here fails closed: a
receipt that cannot be verified raises rather than degrading to a weaker claim.

Refinery recomputes the source bundle hash itself and compares it to the value
the receipt asserts, so a receipt cannot be pointed at a different bundle. It
does not trust the receipt's own account of what it evaluated.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA_ID = "gtdataworks.foundry.evaluation_receipt.v1"
SUPPORTED_SCHEMA_MAJOR = 1

RECEIPT_HASH_FIELD = "receipt_hash"
AUTH_METHOD_HMAC_SHA256 = "hmac-sha256"
TRUST_POLICY_SCHEMA_ID = "goldtrace.refinery.foundry_trust_policy.v1"
MIN_AUTHENTICATION_KEY_BYTES = 32
MAX_AUTHENTICATION_KEY_BYTES = 4096
MAX_TRUST_POLICY_BYTES = 1024 * 1024

_GAP_RULES = frozenset(
    {"max_evidence_gaps", "forbidden_evidence_gap_ids", "allowed_evidence_gap_ids"}
)

#: Identifies which path produced an ingot. Recorded on every ingot so a legacy
#: cast can never be read as evidence that Foundry evaluation occurred.
EVALUATION_PATH_THREE_TIER = "three_tier_labyrinth_foundry_refinery"
EVALUATION_PATH_LEGACY = "legacy_direct_labyrinth_to_refinery"

#: Set to 1/true/yes to make a missing receipt an error everywhere.
REQUIRE_RECEIPT_ENV_VAR = "GOLDTRACE_REQUIRE_FOUNDRY_RECEIPT"


class FoundryReceiptError(ValueError):
    """A Foundry evaluation receipt is absent, malformed, or does not bind."""


@dataclass(frozen=True)
class TrustedEvaluationPack:
    pack_id: str
    pack_version: str
    pack_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.pack_id, str) or not self.pack_id.strip():
            raise FoundryReceiptError("trusted evaluation pack id must be a non-empty string")
        if not isinstance(self.pack_version, str) or not self.pack_version.strip():
            raise FoundryReceiptError("trusted evaluation pack id and version must be non-empty")
        if not _is_sha256(self.pack_hash):
            raise FoundryReceiptError("trusted evaluation pack hash must be lowercase SHA-256")


@dataclass(frozen=True)
class FoundryTrustPolicy:
    """Explicit trusted issuer key and evaluation-pack allowlist."""

    issuer_id: str
    authentication_method: str
    key_id: str
    authentication_key: bytes = field(repr=False)
    evaluation_packs: tuple[TrustedEvaluationPack, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.issuer_id, str) or not self.issuer_id.strip():
            raise FoundryReceiptError("trusted Foundry issuer id must be a non-empty string")
        if self.authentication_method != AUTH_METHOD_HMAC_SHA256:
            raise FoundryReceiptError(
                "trusted Foundry authentication method must be hmac-sha256"
            )
        if not isinstance(self.key_id, str) or not self.key_id.strip():
            raise FoundryReceiptError("trusted Foundry key id must be a non-empty string")
        if not isinstance(self.authentication_key, bytes) or not (
            MIN_AUTHENTICATION_KEY_BYTES
            <= len(self.authentication_key)
            <= MAX_AUTHENTICATION_KEY_BYTES
        ):
            raise FoundryReceiptError(
                "trusted Foundry authentication key must contain between "
                f"{MIN_AUTHENTICATION_KEY_BYTES} and {MAX_AUTHENTICATION_KEY_BYTES} bytes"
            )
        if not self.evaluation_packs:
            raise FoundryReceiptError(
                "trusted Foundry policy must allow at least one evaluation pack"
            )
        if not all(
            isinstance(item, TrustedEvaluationPack) for item in self.evaluation_packs
        ):
            raise FoundryReceiptError(
                "trusted Foundry evaluation_packs must contain TrustedEvaluationPack values"
            )
        identities = {
            (item.pack_id, item.pack_version, item.pack_hash) for item in self.evaluation_packs
        }
        if len(identities) != len(self.evaluation_packs):
            raise FoundryReceiptError("trusted Foundry policy contains duplicate evaluation packs")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj)).hexdigest()


def compute_receipt_hash(receipt: dict[str, Any]) -> str:
    body = {key: value for key, value in receipt.items() if key != RECEIPT_HASH_FIELD}
    return _sha256_json(body)


def receipt_authentication_payload(receipt: dict[str, Any]) -> bytes:
    """Canonical receipt bytes covered by Foundry issuer authentication."""

    body = {key: value for key, value in receipt.items() if key != RECEIPT_HASH_FIELD}
    issuer = body.get("issuer")
    if isinstance(issuer, dict):
        issuer_copy = dict(issuer)
        authentication = issuer_copy.get("authentication")
        if isinstance(authentication, dict):
            authentication_copy = dict(authentication)
            authentication_copy.pop("tag", None)
            issuer_copy["authentication"] = authentication_copy
        body["issuer"] = issuer_copy
    return _canonical_json(body)


def compute_receipt_authentication_tag(
    receipt: dict[str, Any],
    authentication_key: bytes,
) -> str:
    return hmac.new(
        authentication_key,
        receipt_authentication_payload(receipt),
        hashlib.sha256,
    ).hexdigest()


def load_receipt(path: str | Path) -> dict[str, Any]:
    receipt_path = Path(path).expanduser().resolve()
    if not receipt_path.is_file():
        raise FoundryReceiptError(f"Foundry evaluation receipt not found: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FoundryReceiptError(f"receipt is not valid JSON: {receipt_path}: {exc}") from exc
    if not isinstance(receipt, dict):
        raise FoundryReceiptError(f"receipt must be a JSON object: {receipt_path}")
    return receipt


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FoundryReceiptError(message)


def _read_stable_owned_file(
    path: Path,
    *,
    label: str,
    forbidden_permission_bits: int,
    permission_error: str,
    maximum_bytes: int,
) -> bytes:
    """Read a security input through one verified, stable file descriptor."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FoundryReceiptError(
            f"{label} must be a regular non-symlink file: {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise FoundryReceiptError(
            f"{label} must be a regular non-symlink file: {path}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise FoundryReceiptError(
                f"{label} must be a regular non-symlink file: {path}"
            )
        if (metadata.st_dev, metadata.st_ino) != (opened.st_dev, opened.st_ino):
            raise FoundryReceiptError(f"{label} changed while opening: {path}")
        if os.name == "posix":
            if opened.st_uid != os.geteuid():
                raise FoundryReceiptError(
                    f"{label} must be owned by the current user: {path}"
                )
            mode = stat.S_IMODE(opened.st_mode)
            if mode & forbidden_permission_bits:
                raise FoundryReceiptError(
                    f"{label} {permission_error} (got mode {mode:04o}): {path}"
                )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(maximum_bytes + 1)
    except FoundryReceiptError:
        raise
    except OSError as exc:
        raise FoundryReceiptError(f"{label} could not be read: {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if len(payload) > maximum_bytes:
        raise FoundryReceiptError(
            f"{label} exceeds the maximum size of {maximum_bytes} bytes: {path}"
        )
    return payload


def load_foundry_trust_policy(path: str | Path) -> FoundryTrustPolicy:
    """Load explicit issuer/key/pack trust inputs without embedding a secret."""

    policy_path = Path(path).expanduser()
    if not policy_path.is_absolute():
        policy_path = Path.cwd() / policy_path
    raw_policy = _read_stable_owned_file(
        policy_path,
        label="Foundry trust policy",
        forbidden_permission_bits=stat.S_IWGRP | stat.S_IWOTH,
        permission_error="must not be group/other-writable",
        maximum_bytes=MAX_TRUST_POLICY_BYTES,
    )
    try:
        document = json.loads(raw_policy.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FoundryReceiptError(
            f"Foundry trust policy is not valid JSON: {policy_path}: {exc}"
        ) from exc
    _require(isinstance(document, dict), "Foundry trust policy must be a JSON object")
    _require(
        document.get("schema_id") == TRUST_POLICY_SCHEMA_ID,
        f"Foundry trust policy schema_id must be {TRUST_POLICY_SCHEMA_ID}",
    )
    expected_policy_fields = {
        "schema_id",
        "issuer_id",
        "authentication_method",
        "key_id",
        "authentication_key_file",
        "evaluation_packs",
    }
    _require(
        set(document) == expected_policy_fields,
        "Foundry trust policy fields must be exactly "
        f"{sorted(expected_policy_fields)}; got {sorted(document)}",
    )
    issuer_id = document.get("issuer_id")
    authentication_method = document.get("authentication_method")
    key_id = document.get("key_id")
    key_file = document.get("authentication_key_file")
    _require(
        isinstance(issuer_id, str) and bool(issuer_id.strip()),
        "trust policy issuer_id is required",
    )
    _require(
        authentication_method == AUTH_METHOD_HMAC_SHA256,
        "trust policy authentication_method must be hmac-sha256",
    )
    _require(
        isinstance(key_id, str) and bool(key_id.strip()),
        "trust policy key_id is required",
    )
    _require(
        isinstance(key_file, str) and bool(key_file),
        "trust policy authentication_key_file is required",
    )

    key_path = Path(key_file).expanduser()
    if not key_path.is_absolute():
        key_path = policy_path.parent / key_path
    authentication_key = _read_stable_owned_file(
        key_path,
        label="trusted Foundry authentication key",
        forbidden_permission_bits=stat.S_IRWXG | stat.S_IRWXO,
        permission_error="must have owner-only permissions",
        maximum_bytes=MAX_AUTHENTICATION_KEY_BYTES,
    )

    packs_document = document.get("evaluation_packs")
    _require(
        isinstance(packs_document, list) and bool(packs_document),
        "trust policy evaluation_packs must be a non-empty list",
    )
    packs: list[TrustedEvaluationPack] = []
    for entry in packs_document:
        _require(isinstance(entry, dict), "each trusted evaluation pack must be an object")
        _require(
            set(entry) == {"pack_id", "pack_version", "pack_hash"},
            "trusted evaluation pack fields must be exactly pack_id, pack_version and pack_hash",
        )
        packs.append(
            TrustedEvaluationPack(
                pack_id=entry.get("pack_id"),
                pack_version=entry.get("pack_version"),
                pack_hash=entry.get("pack_hash"),
            )
        )
    return FoundryTrustPolicy(
        issuer_id=issuer_id,
        authentication_method=authentication_method,
        key_id=key_id,
        authentication_key=authentication_key,
        evaluation_packs=tuple(packs),
    )


def _verify_evaluation_consistency(receipt: dict[str, Any]) -> str:
    """Independently derive status/counts from the signed check records."""

    evaluation = receipt.get("evaluation")
    _require(isinstance(evaluation, dict), "receipt is missing its evaluation block")
    _require(
        evaluation.get("deterministic") is True,
        "receipt evaluation must declare deterministic=true",
    )
    _require(receipt.get("model") is None, "deterministic Foundry evaluation must not name a model")
    checks = evaluation.get("checks")
    _require(
        isinstance(checks, list) and bool(checks),
        "receipt evaluation checks must be a non-empty list",
    )

    seen: set[str] = set()
    passed = held = failed_required = failed_advisory = errored = 0
    governs_gaps = False
    for index, check in enumerate(checks):
        _require(isinstance(check, dict), f"receipt evaluation check {index} must be an object")
        check_id = check.get("check_id")
        _require(
            isinstance(check_id, str) and bool(check_id) and check_id not in seen,
            f"receipt evaluation check {index} has a missing or duplicate check_id",
        )
        seen.add(check_id)
        rule = check.get("rule")
        severity = check.get("severity")
        check_status = check.get("status")
        _require(isinstance(rule, str) and bool(rule), f"receipt check {check_id} lacks a rule")
        _require(
            severity in {"required", "advisory"},
            f"receipt check {check_id} has invalid severity {severity!r}",
        )
        _require(
            check_status in {"PASS", "HOLD", "FAIL", "ERROR"},
            f"receipt check {check_id} has invalid status {check_status!r}",
        )
        governs_gaps = governs_gaps or rule in _GAP_RULES
        if check_status == "PASS":
            passed += 1
        elif check_status == "HOLD":
            held += 1
        elif check_status == "ERROR":
            errored += 1
        elif severity == "required":
            failed_required += 1
        else:
            failed_advisory += 1

    pack = receipt.get("evaluation_pack")
    _require(isinstance(pack, dict), "receipt is missing its evaluation_pack block")
    _require(
        pack.get("check_count") == len(checks),
        "receipt evaluation pack check_count differs from the signed check records",
    )
    expected_counts = {
        "total": len(checks),
        "passed": passed,
        "failed": failed_required + failed_advisory,
        "failed_required": failed_required,
        "failed_advisory": failed_advisory,
        "errored": errored,
    }
    claimed_counts = evaluation.get("counts")
    # Receipt schema 1.3 added an explicit held count. Retain minor-version
    # compatibility with 1.0--1.2 receipts, which cannot contain HOLD checks
    # and legitimately omit this field.
    if isinstance(claimed_counts, dict) and ("held" in claimed_counts or held):
        expected_counts["held"] = held
    _require(
        isinstance(claimed_counts, dict)
        and set(claimed_counts) == set(expected_counts)
        and all(type(value) is int and value >= 0 for value in claimed_counts.values()),
        "receipt evaluation counts must contain exactly the non-negative integer count fields",
    )
    _require(
        claimed_counts == expected_counts,
        "receipt evaluation counts are inconsistent with the signed check records",
    )

    source = receipt.get("source")
    _require(isinstance(source, dict), "receipt is missing its source block")
    source_gaps = source.get("evidence_gap_ids")
    receipt_gaps = receipt.get("evidence_gaps")
    _require(
        isinstance(source_gaps, list) and isinstance(receipt_gaps, list),
        "receipt evidence gaps must be lists",
    )
    _require(
        all(isinstance(item, str) and bool(item) for item in source_gaps)
        and all(isinstance(item, str) and bool(item) for item in receipt_gaps),
        "receipt evidence gaps must contain non-empty strings",
    )
    _require(
        receipt_gaps == source_gaps,
        "receipt evidence gaps differ from the source evidence-gap declaration",
    )
    ungoverned_gaps = bool(source_gaps) and not governs_gaps

    oracle = receipt.get("oracle")
    oracle_refused = False
    if oracle is not None:
        _require(isinstance(oracle, dict), "receipt oracle block must be an object")
        oracle_status = oracle.get("status")
        _require(
            oracle_status in {"BOUND", "NOT_SUPPLIED", "MISMATCH", "UNBINDABLE"},
            f"receipt oracle block has invalid status {oracle_status!r}",
        )
        oracle_refused = oracle_status in {"MISMATCH", "UNBINDABLE"}
        oracle_rules = {
            "oracle_findings_min_recall",
            "oracle_required_observations_satisfied",
            "oracle_required_report_properties_satisfied",
            "oracle_boundary_controls_respected",
            "oracle_canary_not_propagated",
        }
        oracle_checks = [check for check in checks if check.get("rule") in oracle_rules]
        if "grading_status" in oracle or "graded_rule_count" in oracle:
            _require(
                oracle_status == "BOUND" and bool(oracle_checks),
                "oracle grading disposition requires a BOUND oracle and oracle checks",
            )
            _require(
                oracle.get("graded_rule_count") == len(oracle_checks),
                "oracle graded_rule_count differs from the signed oracle checks",
            )
            derived_oracle_status = (
                "ERROR"
                if any(check.get("status") == "ERROR" for check in oracle_checks)
                else "FAIL"
                if any(check.get("status") == "FAIL" for check in oracle_checks)
                else "HOLD"
                if any(check.get("status") == "HOLD" for check in oracle_checks)
                else "PASS"
            )
            _require(
                oracle.get("grading_status") == derived_oracle_status,
                "oracle grading_status differs from the signed oracle checks",
            )

    if oracle_refused or errored:
        derived_status = "ERROR"
    elif failed_required:
        derived_status = "FAIL"
    elif held or failed_advisory or ungoverned_gaps:
        derived_status = "HOLD"
    else:
        derived_status = "PASS"
    _require(
        receipt.get("status") == derived_status,
        "receipt status is inconsistent with its signed evaluation checks: "
        f"claimed {receipt.get('status')!r}, derived {derived_status!r}",
    )
    return derived_status


def verify_foundry_receipt(
    receipt: dict[str, Any],
    *,
    source_bundle_hash: str,
    source_manifest_hash: str,
    labyrinth_run_id: str | None,
    trust_policy: FoundryTrustPolicy | None = None,
) -> dict[str, Any]:
    """Verify a receipt binds to this bundle and admits it.

    ``source_bundle_hash`` and ``source_manifest_hash`` must be values Refinery
    computed itself. Returns the ``foundry_evaluation`` block to embed in the
    ingot; raises :class:`FoundryReceiptError` on any failure.
    """
    _require(isinstance(receipt, dict), "receipt must be a JSON object")
    _require(
        trust_policy is not None,
        "no trusted Foundry issuer/channel is configured; an unkeyed receipt "
        "checksum cannot authenticate a Foundry decision",
    )

    # --- identity of the receipt format -----------------------------------
    _require(
        receipt.get("schema_id") == RECEIPT_SCHEMA_ID,
        f"unexpected receipt schema_id: {receipt.get('schema_id')!r} "
        f"(expected {RECEIPT_SCHEMA_ID})",
    )
    raw_version = receipt.get("schema_version")
    _require(isinstance(raw_version, str), "receipt is missing schema_version")
    version_match = re.fullmatch(r"([0-9]+)\.([0-9]+)\.([0-9]+)", raw_version)
    _require(
        version_match is not None,
        f"unparseable receipt schema_version: {raw_version!r}; expected MAJOR.MINOR.PATCH",
    )
    version = raw_version
    major = int(version_match.group(1))
    _require(
        major == SUPPORTED_SCHEMA_MAJOR,
        f"unsupported receipt schema major version {major}; this Refinery "
        f"understands {SUPPORTED_SCHEMA_MAJOR}.x",
    )

    # --- integrity of the receipt itself ----------------------------------
    claimed_hash = receipt.get(RECEIPT_HASH_FIELD)
    _require(_is_sha256(claimed_hash), "receipt is missing a well-formed receipt_hash")
    recomputed = compute_receipt_hash(receipt)
    _require(
        hmac.compare_digest(recomputed, claimed_hash),
        "receipt integrity check failed: recomputed hash does not match the "
        "receipt_hash it carries (the receipt was altered after it was issued)",
    )

    # --- authenticated issuer channel --------------------------------------
    issuer = receipt.get("issuer")
    _require(isinstance(issuer, dict), "receipt is missing its issuer block")
    _require(
        issuer.get("issuer_id") == trust_policy.issuer_id,
        "receipt issuer id is not trusted: receipt asserts "
        f"{issuer.get('issuer_id')!r}, policy requires {trust_policy.issuer_id!r}",
    )
    authentication = issuer.get("authentication")
    _require(isinstance(authentication, dict), "receipt is missing issuer authentication")
    _require(
        authentication.get("method") == trust_policy.authentication_method,
        "receipt is not issuer-authenticated with hmac-sha256; no authenticated "
        "Foundry decision is available",
    )
    _require(
        authentication.get("key_id") == trust_policy.key_id,
        "receipt authentication key id is not trusted: receipt asserts "
        f"{authentication.get('key_id')!r}, policy requires {trust_policy.key_id!r}",
    )
    claimed_tag = authentication.get("tag")
    _require(_is_sha256(claimed_tag), "receipt is missing a well-formed authentication tag")
    expected_tag = compute_receipt_authentication_tag(
        receipt,
        trust_policy.authentication_key,
    )
    _require(
        hmac.compare_digest(expected_tag, claimed_tag),
        "receipt issuer authentication failed; refusing a forged or untrusted decision",
    )

    # --- binding to this exact bundle -------------------------------------
    source = receipt.get("source")
    _require(isinstance(source, dict), "receipt is missing its source block")
    _require(
        source.get("class") == "goldtrace_labyrinth_sealed_bundle",
        "receipt source class is not a sealed Labyrinth bundle",
    )
    _require(
        source.get("bundle_hash") == source_bundle_hash,
        "receipt source bundle hash does not match this bundle: receipt asserts "
        f"{source.get('bundle_hash')!r}, Refinery computed {source_bundle_hash!r}",
    )
    _require(
        source.get("manifest_hash") == source_manifest_hash,
        "receipt source manifest hash does not match this bundle: receipt asserts "
        f"{source.get('manifest_hash')!r}, Refinery computed {source_manifest_hash!r}",
    )
    if labyrinth_run_id is not None:
        _require(
            source.get("labyrinth_run_id") == labyrinth_run_id,
            "receipt run id does not match this bundle: receipt asserts "
            f"{source.get('labyrinth_run_id')!r}, bundle is {labyrinth_run_id!r}",
        )

    # --- the evaluation pack must be identified and hashed ----------------
    pack = receipt.get("evaluation_pack")
    _require(isinstance(pack, dict), "receipt is missing its evaluation_pack block")
    pack_hash = pack.get("pack_hash")
    _require(
        _is_sha256(pack_hash),
        "receipt evaluation_pack is missing a well-formed pack_hash",
    )
    _require(
        isinstance(pack.get("pack_id"), str) and bool(pack["pack_id"]),
        "receipt evaluation_pack is missing pack_id",
    )
    _require(
        isinstance(pack.get("pack_version"), str) and bool(pack["pack_version"]),
        "receipt evaluation_pack is missing pack_version",
    )
    pack_identity = (pack.get("pack_id"), pack.get("pack_version"), pack_hash)
    trusted_pack_identities = {
        (item.pack_id, item.pack_version, item.pack_hash)
        for item in trust_policy.evaluation_packs
    }
    _require(
        pack_identity in trusted_pack_identities,
        "receipt evaluation pack identity/hash is not trusted: "
        f"{pack_identity!r}",
    )

    # --- independently derive the signed decision --------------------------
    status = _verify_evaluation_consistency(receipt)

    # --- admission decision ------------------------------------------------
    _require(
        status == "PASS",
        f"Foundry evaluation status {status!r} is not admissible; the three-tier "
        "gate requires PASS. Refusing to cast.",
    )

    foundry = receipt.get("foundry") if isinstance(receipt.get("foundry"), dict) else {}
    evaluation = receipt.get("evaluation") if isinstance(receipt.get("evaluation"), dict) else {}

    return {
        "receipt_schema_id": RECEIPT_SCHEMA_ID,
        "receipt_schema_version": version,
        "receipt_hash": claimed_hash,
        "status": status,
        "evaluation_pack_id": pack.get("pack_id"),
        "evaluation_pack_version": pack.get("pack_version"),
        "evaluation_pack_hash": pack_hash,
        "foundry_package": foundry.get("package"),
        "foundry_version": foundry.get("version"),
        "foundry_code_sha256": foundry.get("handoff_module_sha256"),
        "model_identity": receipt.get("model"),
        "deterministic": bool(evaluation.get("deterministic", False)),
        "evaluation_counts": evaluation.get("counts") or {},
        "evidence_gaps": list(receipt.get("evidence_gaps") or []),
        "limitations": list(receipt.get("limitations") or []),
        "evaluated_at": receipt.get("created_at"),
        "trust": {
            "channel": trust_policy.authentication_method,
            "issuer_id": trust_policy.issuer_id,
            "key_id": trust_policy.key_id,
            "pack_identity_verified": True,
            "evaluation_consistency_verified": True,
        },
    }


def validate_against_foundry_schema(receipt: dict[str, Any]) -> str | None:
    """Additionally validate against Foundry's shipped schema when available.

    Returns ``None`` on success, or a reason string when validation could not be
    performed. Structural verification above is the primary gate; this is a
    second opinion, not a substitute.
    """
    try:
        from evalfoundry.handoff import schemas_dir
    except ImportError as exc:
        return f"evalfoundry not importable: {exc}"
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - jsonschema is a base dependency
        return f"jsonschema not importable: {exc}"

    schema_path = schemas_dir() / "foundry-evaluation-receipt.v1.schema.json"
    if not schema_path.is_file():
        return f"schema not found: {schema_path}"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(receipt)
    return None


def require_receipt_by_default() -> bool:
    """Whether a missing receipt is an error when the caller did not say."""
    import os

    return os.environ.get(REQUIRE_RECEIPT_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}
