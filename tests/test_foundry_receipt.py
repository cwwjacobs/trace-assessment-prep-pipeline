"""Tier-2 gate: Refinery must refuse anything but a receipt that binds to the bundle.

These tests use synthetic receipts so they run without the Foundry or Labyrinth
components. The root three-tier integration suite exercises the same gate with a
real receipt produced by Foundry from a real sealed bundle.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import goldtrace_refinery.cast as cast_module
from goldtrace_refinery.cli import main as cli_main
from goldtrace_refinery.foundry_receipt import (
    AUTH_METHOD_HMAC_SHA256,
    EVALUATION_PATH_LEGACY,
    EVALUATION_PATH_THREE_TIER,
    RECEIPT_SCHEMA_ID,
    TRUST_POLICY_SCHEMA_ID,
    FoundryTrustPolicy,
    FoundryReceiptError,
    TrustedEvaluationPack,
    compute_receipt_authentication_tag,
    compute_receipt_hash,
    load_foundry_trust_policy,
    load_receipt,
    require_receipt_by_default,
    verify_foundry_receipt,
)
from goldtrace_refinery.ingot_schema import validate_ingot
from goldtrace_refinery.verify_bundle import BundleVerifyResult

BUNDLE_HASH = "a" * 64
MANIFEST_HASH = "b" * 64
RUN_ID = "run-20260809t000000z-000000000000"
PACK_HASH = "c" * 64
AUTHENTICATION_KEY = b"test-only-foundry-authentication-key"
ISSUER_ID = "foundry.test"
KEY_ID = "foundry-test-key-1"
TRUST_POLICY = FoundryTrustPolicy(
    issuer_id=ISSUER_ID,
    authentication_method=AUTH_METHOD_HMAC_SHA256,
    key_id=KEY_ID,
    authentication_key=AUTHENTICATION_KEY,
    evaluation_packs=(
        TrustedEvaluationPack(
            pack_id="labyrinth-bundle-admission",
            pack_version="1.0.0",
            pack_hash=PACK_HASH,
        ),
    ),
)


def sign_receipt(
    receipt: dict,
    *,
    authentication_key: bytes = AUTHENTICATION_KEY,
    issuer_id: str = ISSUER_ID,
    key_id: str = KEY_ID,
) -> dict:
    receipt.pop("receipt_hash", None)
    receipt["issuer"] = {
        "issuer_id": issuer_id,
        "authentication": {
            "method": AUTH_METHOD_HMAC_SHA256,
            "key_id": key_id,
        },
    }
    receipt["issuer"]["authentication"]["tag"] = compute_receipt_authentication_tag(
        receipt,
        authentication_key,
    )
    receipt["receipt_hash"] = compute_receipt_hash(receipt)
    return receipt


def make_receipt(**overrides) -> dict:
    status = overrides.get("status", "PASS")
    if status == "HOLD":
        check_status, severity = "FAIL", "advisory"
    elif status == "FAIL":
        check_status, severity = "FAIL", "required"
    elif status == "ERROR":
        check_status, severity = "ERROR", "required"
    else:
        check_status, severity = "PASS", "required"
    checks = [
        {
            "check_id": "c1",
            "rule": "event_chain_intact",
            "severity": severity,
            "status": check_status,
        }
    ]
    counts = {
        "total": 1,
        "passed": int(check_status == "PASS"),
        "failed": int(check_status == "FAIL"),
        "failed_required": int(check_status == "FAIL" and severity == "required"),
        "failed_advisory": int(check_status == "FAIL" and severity == "advisory"),
        "errored": int(check_status == "ERROR"),
    }
    receipt = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": "1.0.0",
        "created_at": "2026-08-09T00:00:00Z",
        "status": status,
        "source": {
            "class": "goldtrace_labyrinth_sealed_bundle",
            "bundle_hash": BUNDLE_HASH,
            "manifest_hash": MANIFEST_HASH,
            "labyrinth_run_id": RUN_ID,
            "evidence_gap_ids": [],
        },
        "evaluation_pack": {
            "pack_id": "labyrinth-bundle-admission",
            "pack_version": "1.0.0",
            "pack_hash": PACK_HASH,
            "check_count": 1,
        },
        "foundry": {
            "package": "evalfoundry",
            "version": "0.3.0",
            "handoff_module_sha256": "d" * 64,
        },
        "model": None,
        "evaluation": {"deterministic": True, "checks": checks, "counts": counts},
        "evidence_gaps": [],
        "limitations": ["No model was invoked."],
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(receipt.get(key), dict):
            receipt[key] = {**receipt[key], **value}
        else:
            receipt[key] = value
    return sign_receipt(receipt)


def verify(receipt: dict, **kwargs):
    return verify_foundry_receipt(
        receipt,
        source_bundle_hash=kwargs.pop("source_bundle_hash", BUNDLE_HASH),
        source_manifest_hash=kwargs.pop("source_manifest_hash", MANIFEST_HASH),
        labyrinth_run_id=kwargs.pop("labyrinth_run_id", RUN_ID),
        trust_policy=kwargs.pop("trust_policy", TRUST_POLICY),
        **kwargs,
    )


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_a_binding_pass_receipt_yields_the_evaluation_block() -> None:
    block = verify(make_receipt())
    assert block["status"] == "PASS"
    assert block["evaluation_pack_hash"] == PACK_HASH
    assert block["evaluation_pack_id"] == "labyrinth-bundle-admission"
    assert len(block["receipt_hash"]) == 64
    assert block["deterministic"] is True
    assert block["trust"] == {
        "channel": AUTH_METHOD_HMAC_SHA256,
        "issuer_id": ISSUER_ID,
        "key_id": KEY_ID,
        "pack_identity_verified": True,
        "evaluation_consistency_verified": True,
    }


# --------------------------------------------------------------------------
# integrity
# --------------------------------------------------------------------------


def test_altered_receipt_fails_the_integrity_check() -> None:
    receipt = make_receipt(status="HOLD")
    receipt["status"] = "PASS"  # flip the decision without reissuing the hash
    with pytest.raises(FoundryReceiptError, match="integrity"):
        verify(receipt)


def test_rehashed_hold_to_pass_forgery_fails_issuer_authentication() -> None:
    receipt = make_receipt(status="HOLD")
    receipt["status"] = "PASS"
    receipt["receipt_hash"] = compute_receipt_hash(receipt)
    assert receipt["receipt_hash"] == compute_receipt_hash(receipt)
    with pytest.raises(FoundryReceiptError, match="issuer authentication failed"):
        verify(receipt)


def test_receipt_without_explicit_trust_policy_is_rejected() -> None:
    with pytest.raises(FoundryReceiptError, match="no trusted Foundry issuer/channel"):
        verify(make_receipt(), trust_policy=None)


def test_unsigned_receipt_is_rejected() -> None:
    receipt = make_receipt()
    receipt["issuer"] = {
        "issuer_id": ISSUER_ID,
        "authentication": {"method": "none", "reason": "no key configured"},
    }
    receipt["receipt_hash"] = compute_receipt_hash(receipt)
    with pytest.raises(FoundryReceiptError, match="not issuer-authenticated"):
        verify(receipt)


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("issuer", "issuer id is not trusted"),
        ("key_id", "key id is not trusted"),
        ("key", "issuer authentication failed"),
    ],
)
def test_wrong_issuer_or_key_is_rejected(mutation: str, pattern: str) -> None:
    receipt = make_receipt()
    if mutation == "issuer":
        sign_receipt(receipt, issuer_id="foundry.attacker")
    elif mutation == "key_id":
        sign_receipt(receipt, key_id="attacker-key")
    else:
        sign_receipt(receipt, authentication_key=b"attacker-controlled-authentication-key")
    with pytest.raises(FoundryReceiptError, match=pattern):
        verify(receipt)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pack_id", "attacker-pack"),
        ("pack_version", "9.9.9"),
        ("pack_hash", "e" * 64),
    ],
)
def test_trusted_issuer_cannot_use_an_untrusted_pack_identity(
    field: str,
    value: str,
) -> None:
    receipt = make_receipt(evaluation_pack={field: value})
    sign_receipt(receipt)
    with pytest.raises(FoundryReceiptError, match="pack identity/hash is not trusted"):
        verify(receipt)


@pytest.mark.parametrize("mutation", ["status", "counts", "checks"])
def test_signed_receipt_must_be_internally_consistent(mutation: str) -> None:
    receipt = make_receipt()
    if mutation == "status":
        receipt["status"] = "HOLD"
    elif mutation == "counts":
        receipt["evaluation"]["counts"]["passed"] = 0
    else:
        receipt["evaluation"]["checks"][0]["status"] = "FAIL"
    sign_receipt(receipt)
    with pytest.raises(FoundryReceiptError, match="inconsistent"):
        verify(receipt)


@pytest.mark.parametrize("invalid_count", [True, 1.0, -1])
def test_signed_receipt_counts_must_be_non_negative_integers(invalid_count) -> None:
    receipt = make_receipt()
    receipt["evaluation"]["counts"]["passed"] = invalid_count
    sign_receipt(receipt)
    with pytest.raises(FoundryReceiptError, match="non-negative integer"):
        verify(receipt)


def test_missing_receipt_hash_is_rejected() -> None:
    receipt = make_receipt()
    del receipt["receipt_hash"]
    with pytest.raises(FoundryReceiptError, match="receipt_hash"):
        verify(receipt)


def test_truncated_receipt_hash_is_rejected() -> None:
    receipt = make_receipt()
    receipt["receipt_hash"] = "abc123"
    with pytest.raises(FoundryReceiptError, match="receipt_hash"):
        verify(receipt)


# --------------------------------------------------------------------------
# binding to this bundle
# --------------------------------------------------------------------------


def test_receipt_for_a_different_bundle_is_rejected() -> None:
    with pytest.raises(FoundryReceiptError, match="bundle hash"):
        verify(make_receipt(), source_bundle_hash="f" * 64)


def test_receipt_with_a_different_manifest_hash_is_rejected() -> None:
    with pytest.raises(FoundryReceiptError, match="manifest hash"):
        verify(make_receipt(), source_manifest_hash="f" * 64)


def test_receipt_for_a_different_run_is_rejected() -> None:
    with pytest.raises(FoundryReceiptError, match="run id"):
        verify(make_receipt(), labyrinth_run_id="run-someone-elses")


# --------------------------------------------------------------------------
# unacceptable states fail closed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,pattern",
    [
        ("HOLD", "not admissible"),
        ("FAIL", "not admissible"),
        ("ERROR", "not admissible"),
        ("UNKNOWN", "inconsistent"),
        ("", "inconsistent"),
    ],
)
def test_non_pass_statuses_fail_closed(status: str, pattern: str) -> None:
    with pytest.raises(FoundryReceiptError, match=pattern):
        verify(make_receipt(status=status))


def test_explicit_held_check_is_derived_and_refused_not_misparsed() -> None:
    receipt = make_receipt(schema_version="1.3.0")
    receipt["status"] = "HOLD"
    receipt["evaluation"]["checks"][0].update(
        {
            "rule": "oracle_required_observations_satisfied",
            "severity": "required",
            "status": "HOLD",
            "replay_seam": "agent_observation_coverage",
        }
    )
    receipt["evaluation"]["counts"] = {
        "total": 1,
        "passed": 0,
        "failed": 0,
        "failed_required": 0,
        "failed_advisory": 0,
        "held": 1,
        "errored": 0,
    }
    receipt["oracle"] = {
        "status": "BOUND",
        "grading_status": "HOLD",
        "graded_rule_count": 1,
    }
    sign_receipt(receipt)

    with pytest.raises(FoundryReceiptError, match="status 'HOLD' is not admissible"):
        verify(receipt)


def test_oracle_binding_refusal_is_independently_derived_as_error() -> None:
    receipt = make_receipt(schema_version="1.2.0")
    receipt["status"] = "ERROR"
    receipt["oracle"] = {"status": "MISMATCH"}
    sign_receipt(receipt)

    with pytest.raises(FoundryReceiptError, match="status 'ERROR' is not admissible"):
        verify(receipt)


def test_oracle_grading_summary_must_match_its_signed_checks() -> None:
    receipt = make_receipt(schema_version="1.3.0")
    receipt["oracle"] = {
        "status": "BOUND",
        "grading_status": "FAIL",
        "graded_rule_count": 1,
    }
    receipt["evaluation"]["checks"][0]["rule"] = "oracle_findings_min_recall"
    receipt["evaluation"]["counts"]["held"] = 0
    sign_receipt(receipt)

    with pytest.raises(FoundryReceiptError, match="grading_status"):
        verify(receipt)


def test_production_interfaces_expose_no_status_widening_override(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "accepted_statuses" not in inspect.signature(verify_foundry_receipt).parameters
    assert (
        "accepted_receipt_statuses"
        not in inspect.signature(cast_module.cast_mine_bundle).parameters
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_main(
            [
                "cast",
                "/sealed/bundle",
                "--out",
                "/outside/output",
                "--accept-status",
                "HOLD",
            ]
        )
    assert excinfo.value.code == 2
    assert "unrecognized arguments: --accept-status HOLD" in capsys.readouterr().err


# --------------------------------------------------------------------------
# format identity
# --------------------------------------------------------------------------


def test_foreign_schema_id_is_rejected() -> None:
    with pytest.raises(FoundryReceiptError, match="schema_id"):
        verify(make_receipt(schema_id="some.other.receipt.v1"))


def test_future_major_schema_version_is_rejected() -> None:
    with pytest.raises(FoundryReceiptError, match="unsupported receipt schema major"):
        verify(make_receipt(schema_version="2.0.0"))


def test_minor_schema_version_is_accepted() -> None:
    assert verify(make_receipt(schema_version="1.7.3"))["status"] == "PASS"


@pytest.mark.parametrize(
    "version",
    [None, 1, "1", "1.0", "1.0.", "1.0.0-extra", "1.0.0.1", " 1.0.0"],
)
def test_receipt_schema_version_requires_three_numeric_components(version) -> None:
    with pytest.raises(FoundryReceiptError, match="schema_version"):
        verify(make_receipt(schema_version=version))


def test_receipt_without_a_pack_hash_is_rejected() -> None:
    receipt = make_receipt()
    receipt["evaluation_pack"] = {
        "pack_id": "p",
        "pack_version": "1.0.0",
        "check_count": 1,
    }
    sign_receipt(receipt)
    with pytest.raises(FoundryReceiptError, match="pack_hash"):
        verify(receipt)


def test_receipt_without_a_source_block_is_rejected() -> None:
    receipt = make_receipt()
    del receipt["source"]
    sign_receipt(receipt)
    with pytest.raises(FoundryReceiptError, match="source block"):
        verify(receipt)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_absent_receipt_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FoundryReceiptError, match="not found"):
        load_receipt(tmp_path / "nope.json")


def test_malformed_receipt_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(FoundryReceiptError, match="not valid JSON"):
        load_receipt(path)


def test_non_object_receipt_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(FoundryReceiptError, match="JSON object"):
        load_receipt(path)


def test_trust_policy_loads_external_owner_only_key_and_pack_pins(tmp_path: Path) -> None:
    key_path = tmp_path / "foundry.key"
    key_path.write_bytes(AUTHENTICATION_KEY)
    key_path.chmod(0o600)
    policy_path = tmp_path / "foundry-trust.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_id": TRUST_POLICY_SCHEMA_ID,
                "issuer_id": ISSUER_ID,
                "authentication_method": AUTH_METHOD_HMAC_SHA256,
                "key_id": KEY_ID,
                "authentication_key_file": key_path.name,
                "evaluation_packs": [
                    {
                        "pack_id": "labyrinth-bundle-admission",
                        "pack_version": "1.0.0",
                        "pack_hash": PACK_HASH,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy_path.chmod(0o600)
    assert load_foundry_trust_policy(policy_path) == TRUST_POLICY


def test_trust_policy_document_matches_the_shipped_schema(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    document = {
        "schema_id": TRUST_POLICY_SCHEMA_ID,
        "issuer_id": ISSUER_ID,
        "authentication_method": AUTH_METHOD_HMAC_SHA256,
        "key_id": KEY_ID,
        "authentication_key_file": "foundry.key",
        "evaluation_packs": [
            {
                "pack_id": "labyrinth-bundle-admission",
                "pack_version": "1.0.0",
                "pack_hash": PACK_HASH,
            }
        ],
    }
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "goldtrace_refinery"
        / "contracts"
        / "foundry-trust-policy.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(document)


def test_trust_policy_rejects_exposed_or_missing_key_material(tmp_path: Path) -> None:
    key_path = tmp_path / "foundry.key"
    key_path.write_bytes(AUTHENTICATION_KEY)
    key_path.chmod(0o644)
    policy_path = tmp_path / "foundry-trust.json"
    document = {
        "schema_id": TRUST_POLICY_SCHEMA_ID,
        "issuer_id": ISSUER_ID,
        "authentication_method": AUTH_METHOD_HMAC_SHA256,
        "key_id": KEY_ID,
        "authentication_key_file": key_path.name,
        "evaluation_packs": [
            {
                "pack_id": "labyrinth-bundle-admission",
                "pack_version": "1.0.0",
                "pack_hash": PACK_HASH,
            }
        ],
    }
    policy_path.write_text(json.dumps(document), encoding="utf-8")
    policy_path.chmod(0o600)
    with pytest.raises(FoundryReceiptError, match="owner-only permissions"):
        load_foundry_trust_policy(policy_path)

    document["authentication_key_file"] = "absent.key"
    policy_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(FoundryReceiptError, match="regular non-symlink"):
        load_foundry_trust_policy(policy_path)


@pytest.mark.parametrize("unsafe_kind", ["symlink", "writable"])
def test_trust_policy_file_itself_must_be_secure(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    key_path = tmp_path / "foundry.key"
    key_path.write_bytes(AUTHENTICATION_KEY)
    key_path.chmod(0o600)
    document = {
        "schema_id": TRUST_POLICY_SCHEMA_ID,
        "issuer_id": ISSUER_ID,
        "authentication_method": AUTH_METHOD_HMAC_SHA256,
        "key_id": KEY_ID,
        "authentication_key_file": key_path.name,
        "evaluation_packs": [
            {
                "pack_id": "labyrinth-bundle-admission",
                "pack_version": "1.0.0",
                "pack_hash": PACK_HASH,
            }
        ],
    }
    target = tmp_path / "policy-target.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    target.chmod(0o600)
    policy_path = target
    if unsafe_kind == "symlink":
        policy_path = tmp_path / "policy-link.json"
        policy_path.symlink_to(target)
        match = "regular non-symlink"
    else:
        target.chmod(0o666)
        match = "must not be group/other-writable"

    with pytest.raises(FoundryReceiptError, match=match):
        load_foundry_trust_policy(policy_path)


def test_trust_policy_rejects_descriptor_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import goldtrace_refinery.foundry_receipt as receipt_module

    policy_path = tmp_path / "foundry-trust.json"
    policy_path.write_text("{}", encoding="utf-8")
    policy_path.chmod(0o600)
    opened = policy_path.stat()
    monkeypatch.setattr(
        receipt_module.os,
        "fstat",
        lambda _descriptor: SimpleNamespace(
            st_mode=opened.st_mode,
            st_dev=opened.st_dev,
            st_ino=opened.st_ino + 1,
            st_uid=opened.st_uid,
        ),
    )

    with pytest.raises(FoundryReceiptError, match="changed while opening"):
        load_foundry_trust_policy(policy_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership contract")
def test_trust_policy_must_be_owned_by_the_current_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import goldtrace_refinery.foundry_receipt as receipt_module

    policy_path = tmp_path / "foundry-trust.json"
    policy_path.write_text("{}", encoding="utf-8")
    policy_path.chmod(0o600)
    current_uid = os.geteuid()
    monkeypatch.setattr(receipt_module.os, "geteuid", lambda: current_uid + 1)

    with pytest.raises(FoundryReceiptError, match="owned by the current user"):
        load_foundry_trust_policy(policy_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership contract")
def test_trust_policy_key_must_be_owned_by_the_current_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import goldtrace_refinery.foundry_receipt as receipt_module

    key_path = tmp_path / "foundry.key"
    key_path.write_bytes(AUTHENTICATION_KEY)
    key_path.chmod(0o600)
    policy_path = tmp_path / "foundry-trust.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_id": TRUST_POLICY_SCHEMA_ID,
                "issuer_id": ISSUER_ID,
                "authentication_method": AUTH_METHOD_HMAC_SHA256,
                "key_id": KEY_ID,
                "authentication_key_file": key_path.name,
                "evaluation_packs": [
                    {
                        "pack_id": "labyrinth-bundle-admission",
                        "pack_version": "1.0.0",
                        "pack_hash": PACK_HASH,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy_path.chmod(0o600)
    real_fstat = os.fstat
    calls = 0

    def foreign_key_owner(descriptor: int):
        nonlocal calls
        calls += 1
        opened = real_fstat(descriptor)
        if calls == 1:
            return opened
        return SimpleNamespace(
            st_mode=opened.st_mode,
            st_dev=opened.st_dev,
            st_ino=opened.st_ino,
            st_uid=opened.st_uid + 1,
        )

    monkeypatch.setattr(receipt_module.os, "fstat", foreign_key_owner)
    with pytest.raises(FoundryReceiptError, match="owned by the current user"):
        load_foundry_trust_policy(policy_path)


def test_trust_policy_requires_an_explicit_supported_authentication_method(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "foundry.key"
    key_path.write_bytes(AUTHENTICATION_KEY)
    key_path.chmod(0o600)
    policy_path = tmp_path / "foundry-trust.json"
    document = {
        "schema_id": TRUST_POLICY_SCHEMA_ID,
        "issuer_id": ISSUER_ID,
        "authentication_method": "none",
        "key_id": KEY_ID,
        "authentication_key_file": key_path.name,
        "evaluation_packs": [
            {
                "pack_id": "labyrinth-bundle-admission",
                "pack_version": "1.0.0",
                "pack_hash": PACK_HASH,
            }
        ],
    }
    policy_path.write_text(json.dumps(document), encoding="utf-8")
    policy_path.chmod(0o600)
    with pytest.raises(FoundryReceiptError, match="authentication_method"):
        load_foundry_trust_policy(policy_path)


@pytest.mark.parametrize("rehash_forgery", [False, True])
def test_foundry_gate_rejection_precedes_all_derivative_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rehash_forgery: bool,
) -> None:
    """A HOLD or forged PASS must not even create the requested output directory."""

    bundle_root = tmp_path / "sealed-bundle"
    bundle_root.mkdir()
    out_dir = tmp_path / "cast"
    receipt = make_receipt(status="HOLD")
    if rehash_forgery:
        receipt["status"] = "PASS"
        receipt["receipt_hash"] = compute_receipt_hash(receipt)

    monkeypatch.setattr(cast_module, "sha256_tree", lambda _root: BUNDLE_HASH)
    monkeypatch.setattr(
        cast_module,
        "verify_mine_bundle",
        lambda _root: BundleVerifyResult(
            ok=True,
            run_id=RUN_ID,
            seal_status="SEALED",
            manifest_sha256=MANIFEST_HASH,
            event_count=1,
            evidence_gap_ids=[],
            details={},
        ),
    )
    monkeypatch.setattr(cast_module, "load_json", lambda _path: {"run_id": RUN_ID})

    failure = "issuer authentication failed" if rehash_forgery else "not admissible"
    with pytest.raises(FoundryReceiptError, match=failure):
        cast_module.cast_mine_bundle(
            bundle_root,
            out_dir,
            foundry_receipt=receipt,
            foundry_trust_policy=TRUST_POLICY,
        )
    assert not out_dir.exists()


@pytest.mark.parametrize("destination", ["same", "descendant"])
def test_cast_output_must_be_outside_the_sealed_bundle(
    tmp_path: Path,
    destination: str,
) -> None:
    bundle_root = tmp_path / "sealed-bundle"
    bundle_root.mkdir()
    source_file = bundle_root / "source-evidence.txt"
    source_file.write_text("immutable evidence", encoding="utf-8")
    before = cast_module.sha256_tree(bundle_root)
    out_dir = bundle_root if destination == "same" else bundle_root / "derivatives"

    with pytest.raises(ValueError, match="outside the sealed bundle"):
        cast_module.cast_mine_bundle(bundle_root, out_dir)

    assert cast_module.sha256_tree(bundle_root) == before
    assert source_file.read_text(encoding="utf-8") == "immutable evidence"
    assert not (bundle_root / "normalized_events.jsonl").exists()
    assert not (bundle_root / "normalized_events.redacted.jsonl").exists()
    if destination == "descendant":
        assert not out_dir.exists()


def _minimal_ingot(evaluation_path: str, foundry_evaluation) -> dict:
    return {
        "schema_id": "goldtrace.refinery.ingot.v1",
        "ingot_id": "GTI-unit-test",
        "source_class": "goldtrace_mine_bundle",
        "source_bundle_hash": BUNDLE_HASH,
        "source_manifest_hash": MANIFEST_HASH,
        "evaluation_path": evaluation_path,
        "foundry_evaluation": foundry_evaluation,
        "mechanical_run_status": "COMPLETED",
        "seal_status": "SEALED",
        "normalized_event_stream_hash": "e" * 64,
        "deterministic_results": {},
        "redaction": {},
        "quarantine": {},
        "dedup": {"exact_id": "exact", "structural_id": "structural"},
        "lineage_parents": [],
        "refinery_receipt_hash": "f" * 64,
        "refinery_status": "PASSED",
    }


def test_ingot_schema_couples_each_path_to_its_evaluation_evidence() -> None:
    foundry_block = verify(make_receipt())
    assert validate_ingot(
        _minimal_ingot(EVALUATION_PATH_THREE_TIER, foundry_block)
    ) == []
    assert validate_ingot(_minimal_ingot(EVALUATION_PATH_LEGACY, None)) == []


@pytest.mark.parametrize("mismatch", ["three-tier-null", "three-tier-hold", "legacy-foundry"])
def test_ingot_schema_rejects_path_evaluation_mismatches(mismatch: str) -> None:
    foundry_block = verify(make_receipt())
    if mismatch == "three-tier-null":
        ingot = _minimal_ingot(EVALUATION_PATH_THREE_TIER, None)
    elif mismatch == "three-tier-hold":
        ingot = _minimal_ingot(
            EVALUATION_PATH_THREE_TIER,
            {**foundry_block, "status": "HOLD"},
        )
    else:
        ingot = _minimal_ingot(EVALUATION_PATH_LEGACY, foundry_block)

    assert any("schema:" in error for error in validate_ingot(ingot))


# --------------------------------------------------------------------------
# the requirement switch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("yes", True),
                                            ("on", True), ("0", False), ("", False), ("no", False)])
def test_environment_switch_controls_the_default(monkeypatch, value: str, expected: bool) -> None:
    monkeypatch.setenv("GOLDTRACE_REQUIRE_FOUNDRY_RECEIPT", value)
    assert require_receipt_by_default() is expected


def test_environment_switch_absent_defaults_to_not_required(monkeypatch) -> None:
    monkeypatch.delenv("GOLDTRACE_REQUIRE_FOUNDRY_RECEIPT", raising=False)
    assert require_receipt_by_default() is False


def test_evaluation_path_labels_are_distinct_and_explicit() -> None:
    assert EVALUATION_PATH_THREE_TIER != EVALUATION_PATH_LEGACY
    assert "legacy" in EVALUATION_PATH_LEGACY
    # The legacy label must not imply Foundry participated.
    assert "foundry" not in EVALUATION_PATH_LEGACY.lower()
