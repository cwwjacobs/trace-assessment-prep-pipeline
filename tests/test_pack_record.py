"""The training-ready gate must refuse on each blocking condition.

Admission is an allow-list, so the tests that matter most are the ones proving
an *unrecognised* value blocks. A deny-list would let a new vocabulary term, a
typo or an unmigrated record through simply because nobody had named it as bad.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from goldtrace_refinery import pack_record as pr

COMPONENT_ROOT = Path(__file__).resolve().parents[1]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _rights_assertion() -> dict[str, str]:
    assertion = {
        "asserted_by": "operator-local-test-key",
        "asserted_at": "2026-08-12T00:00:00Z",
        "basis": "operator-authored synthetic fixture",
    }
    assertion["assertion_sha256"] = pr.rights_assertion_sha256(assertion)
    return assertion


def make_record(**overrides) -> dict:
    record = {
        "schema_id": pr.SCHEMA_ID,
        "record_id": "REC-0001",
        "content_sha256": _hash("released content"),
        "released_row_sha256": _hash("exact released row bytes"),
        "released_row_format": "json-canonical-v1",
        "authorship_class": pr.OPERATOR_MATERIAL,
        "rights_status": "OPERATOR_OWNED_FULL_RIGHTS",
        "rights_assertion": _rights_assertion(),
        "privacy_status": "PASS_AUTOMATED_SCAN",
        "privacy_receipt_sha256": _hash("final-row privacy receipt"),
        "privacy_receipt_ref": "privacy/REC-0001.receipt.json",
        "privacy_ruleset_sha256": _hash("privacy ruleset"),
        "quality_review": "AUTOMATED_REVIEW_PASS",
        "derivation_method": "automated_source_derived_transformation",
        "split": pr.TRAIN,
        "split_group_id": "GRP-1",
        "split_group_ids": ["GRP-1"],
        "source_artifacts": [
            {"role": "source_event", "sha256": _hash("original source artifact")}
        ],
        "source": {
            "source_hash_sha256": _hash("original source"),
            "source_bundle_hash": "a" * 64,
            "source_manifest_hash": "b" * 64,
            "mine_run_id": "run-20260809t000000z-abc",
            "ingot_id": "GTI-abc123",
            "foundry_receipt_hash": "c" * 64,
            "evaluation_pack_hash": "d" * 64,
            "source_ingot_status": "PASSED",
            "source_foundry_status": "PASS",
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(record.get(key), dict):
            record[key] = {**record[key], **value}
        else:
            record[key] = value
    if "split_group_id" in overrides and "split_group_ids" not in overrides:
        record["split_group_ids"] = [overrides["split_group_id"]]
    if record.get("schema_id") == pr.V3_SCHEMA_ID:
        assertion = record.get("rights_assertion")
        digest = None
        if isinstance(assertion, dict):
            digest = assertion.get("assertion_sha256")
        binding = record.get("rights_assertion_binding")
        if not isinstance(binding, dict):
            if not isinstance(digest, str):
                digest = _rights_assertion()["assertion_sha256"]
            record["rights_assertion_binding"] = {
                "assertion_sha256": digest,
                "disposition": pr.ASSERTION_DISPOSITION_EXTERNAL,
            }
        record["rights_assertion"] = None
    return record


def make_v1_record(**overrides) -> dict:
    record = make_record(schema_id=pr.V1_SCHEMA_ID)
    for key in (
        "released_row_sha256",
        "released_row_format",
        "privacy_receipt_sha256",
        "privacy_receipt_ref",
        "privacy_ruleset_sha256",
        "split_group_ids",
        "source_artifacts",
        "rights_assertion",
    ):
        record.pop(key)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(record.get(key), dict):
            record[key] = {**record[key], **value}
        else:
            record[key] = value
    return record


# --------------------------------------------------------------------------
# the contract itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [pr.V1_CONTRACT_FILENAME, pr.V2_CONTRACT_FILENAME, pr.V3_CONTRACT_FILENAME],
)
def test_shipped_contract_matches_the_packaged_copy(filename: str) -> None:
    canonical = (COMPONENT_ROOT / "contracts" / filename).read_bytes()
    packaged = (
        COMPONENT_ROOT / "src" / "goldtrace_refinery" / "contracts" / filename
    ).read_bytes()
    assert packaged == canonical, "packaged pack-record contract has drifted"


def test_v2_is_the_producer_default_and_both_versions_are_named() -> None:
    assert pr.SCHEMA_ID == pr.V2_SCHEMA_ID
    assert pr.CONTRACT_FILENAME == pr.V2_CONTRACT_FILENAME
    assert pr.SUPPORTED_SCHEMA_IDS == {pr.V1_SCHEMA_ID, pr.V2_SCHEMA_ID, pr.V3_SCHEMA_ID}


@pytest.mark.parametrize("schema_id", [pr.V1_SCHEMA_ID, pr.V2_SCHEMA_ID, pr.V3_SCHEMA_ID])
def test_each_supported_schema_is_valid_draft_2020_12(schema_id: str) -> None:
    import jsonschema

    jsonschema.Draft202012Validator.check_schema(pr.load_schema(schema_id))


def test_a_well_formed_record_validates() -> None:
    assert pr.validate_record(make_record()) == []


def test_a_legacy_v1_record_keeps_its_original_requirements() -> None:
    record = make_v1_record()
    assert pr.validate_record(record) == []
    assert pr.admit_to_training_ready(record).admitted


@pytest.mark.parametrize(
    "field",
    [
        "released_row_sha256",
        "released_row_format",
        "privacy_receipt_sha256",
        "privacy_receipt_ref",
        "privacy_ruleset_sha256",
        "split_group_ids",
        "source_artifacts",
        "rights_assertion",
    ],
)
def test_v2_release_bindings_are_required(field: str) -> None:
    record = make_record()
    del record[field]
    assert pr.validate_record(record)


def test_v2_admissible_rights_require_an_integrity_bound_assertion() -> None:
    missing = make_record(rights_assertion=None)
    assert pr.validate_record(missing)
    assert not pr.admit_to_training_ready(missing).admitted

    tampered = make_record()
    tampered["rights_assertion"]["basis"] = "changed after assertion"
    assert pr.validate_record(tampered)
    assert not pr.admit_to_training_ready(tampered).admitted


def test_rights_assertion_hash_uses_only_the_canonical_assertion_body() -> None:
    assertion = _rights_assertion()
    body = {key: assertion[key] for key in ("asserted_by", "asserted_at", "basis")}
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert assertion["assertion_sha256"] == expected


@pytest.mark.parametrize(
    "field", ["released_row_sha256", "privacy_receipt_sha256", "privacy_ruleset_sha256"]
)
@pytest.mark.parametrize("invalid_hash", ["f" * 63, "F" * 64, "not-a-hash"])
def test_v2_release_binding_hashes_are_exact_sha256_shapes(
    field: str, invalid_hash: str
) -> None:
    problems = pr.validate_record(make_record(**{field: invalid_hash}))
    assert any(field in problem and "SHA-256" in problem for problem in problems)


def test_v2_privacy_safe_source_locators_are_optional_but_supported() -> None:
    record = make_record(
        source={
            "source_provider": "agy",
            "source_session_id_sha256": _hash("session-1"),
            "native_artifact_sha256": _hash("native artifact"),
            "native_record_ordinal": 42,
            "native_record_sha256": _hash("native row"),
        }
    )
    assert pr.validate_record(record) == []


@pytest.mark.parametrize(
    "private_locator",
    [
        {"source_session_id": "raw-provider-session-id"},
        {"native_record_locator": "private/member/path.jsonl#line=42"},
    ],
)
def test_v2_rejects_raw_native_identifiers_and_paths(private_locator: dict[str, object]) -> None:
    record = make_record(source={**make_record()["source"], **private_locator})
    assert pr.validate_record(record)


@pytest.mark.parametrize(
    "source_override",
    [
        {"source_ingot_status": None},
        {"source_ingot_status": "HOLD"},
        {"source_foundry_status": None},
        {"source_foundry_status": "HOLD"},
        {"foundry_receipt_hash": None},
    ],
)
def test_v2_requires_explicit_passed_ingot_and_authenticated_foundry_source(
    source_override: dict[str, object],
) -> None:
    record = make_record(source=source_override)
    assert pr.validate_record(record)
    assert not pr.admit_to_training_ready(record).admitted


def test_v2_source_artifacts_are_nonempty_exact_hash_bindings() -> None:
    assert pr.validate_record(make_record(source_artifacts=[]))
    assert pr.validate_record(
        make_record(source_artifacts=[{"role": "source_event", "sha256": "short"}])
    )


def test_v2_source_artifact_entries_must_be_unique() -> None:
    artifact = {"role": "source_event", "sha256": _hash("source")}
    assert pr.validate_record(make_record(source_artifacts=[artifact, artifact]))


def test_v2_split_lineage_is_nonempty_unique_and_contains_the_primary() -> None:
    assert pr.validate_record(make_record(split_group_ids=[]))
    assert pr.validate_record(make_record(split_group_ids=["GRP-1", "GRP-1"]))
    problems = pr.validate_record(make_record(split_group_ids=["SESSION-1"]))
    assert any("must contain split_group_id" in problem for problem in problems)


def test_record_without_upward_binding_is_invalid() -> None:
    record = make_record()
    record["source"] = {}
    problems = pr.validate_record(record)
    assert any("source_hash_sha256" in p for p in problems)


def test_unknown_authorship_class_is_invalid() -> None:
    assert pr.validate_record(make_record(authorship_class="vibes"))


def test_foreign_schema_id_is_invalid() -> None:
    assert pr.validate_record(make_record(schema_id="something.else.v1"))


def test_non_string_schema_id_fails_closed_without_raising() -> None:
    assert pr.validate_record(make_record(schema_id=[pr.V2_SCHEMA_ID]))


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_a_clean_record_is_admitted() -> None:
    decision = pr.admit_to_training_ready(make_record())
    assert decision.admitted, decision.reasons
    assert set(decision.checks.values()) == {"PASS"}


# --------------------------------------------------------------------------
# condition 1: rights
# --------------------------------------------------------------------------


def test_unknown_rights_blocks_and_is_the_default_posture() -> None:
    decision = pr.admit_to_training_ready(make_record(rights_status=pr.RIGHTS_UNKNOWN))
    assert not decision.admitted
    assert any("not admissible" in r for r in decision.reasons)
    assert pr.RIGHTS_UNKNOWN not in pr.ADMISSIBLE_RIGHTS_STATUS


def test_an_unrecognised_rights_value_blocks() -> None:
    """Allow-list, not deny-list: a term nobody declared admissible must block."""
    decision = pr.admit_to_training_ready(make_record(rights_status="PROBABLY_FINE_I_THINK"))
    assert not decision.admitted
    assert decision.checks["rights"] == "BLOCKED"


def test_the_transformed_release_posture_is_admissible() -> None:
    record = make_record(
        rights_status="TRANSFORMED_RELEASE_TEXT_ONLY_SOURCE_MATERIAL_EXCLUDED",
        authorship_class=pr.MODEL_PROSE,
    )
    assert pr.admit_to_training_ready(record).admitted


def test_an_operator_may_widen_the_rights_allow_list_explicitly() -> None:
    record = make_record(rights_status="NEGOTIATED_VENDOR_TERMS_2026")
    assert not pr.admit_to_training_ready(record).admitted
    widened = pr.admit_to_training_ready(
        record, admissible_rights={*pr.ADMISSIBLE_RIGHTS_STATUS, "NEGOTIATED_VENDOR_TERMS_2026"}
    )
    assert widened.admitted


def test_an_explicit_empty_rights_allow_list_releases_nothing() -> None:
    decision = pr.admit_to_training_ready(make_record(), admissible_rights=set())
    assert not decision.admitted
    assert decision.checks["rights"] == "BLOCKED"


# --------------------------------------------------------------------------
# condition 2: authorship
# --------------------------------------------------------------------------


def test_mixed_authorship_blocks() -> None:
    decision = pr.admit_to_training_ready(make_record(authorship_class=pr.MIXED))
    assert not decision.admitted
    assert any("split the record" in r for r in decision.reasons)


@pytest.mark.parametrize(
    "authorship",
    [pr.OPERATOR_INPUT, pr.OPERATOR_MATERIAL, pr.MACHINE_RESULT, pr.MODEL_PROSE, pr.MODEL_TOOL_CALL],
)
def test_single_class_records_are_admissible(authorship: str) -> None:
    assert pr.admit_to_training_ready(make_record(authorship_class=authorship)).admitted


def test_provider_authored_classes_are_named() -> None:
    assert pr.PROVIDER_AUTHORED == {pr.MODEL_PROSE, pr.MODEL_REASONING, pr.MODEL_TOOL_CALL}
    assert pr.MACHINE_RESULT not in pr.PROVIDER_AUTHORED
    assert pr.OPERATOR_INPUT not in pr.PROVIDER_AUTHORED


# --------------------------------------------------------------------------
# reasoning policy
# --------------------------------------------------------------------------


def test_reasoning_is_excluded_by_default() -> None:
    decision = pr.admit_to_training_ready(make_record(authorship_class=pr.MODEL_REASONING))
    assert not decision.admitted
    assert any("reasoning policy" in r for r in decision.reasons)


def test_reasoning_can_be_allowed_explicitly() -> None:
    decision = pr.admit_to_training_ready(
        make_record(authorship_class=pr.MODEL_REASONING), reasoning_policy=pr.REASONING_ALLOW
    )
    assert decision.admitted


def test_an_unknown_reasoning_policy_is_an_error_not_a_default() -> None:
    with pytest.raises(pr.PackRecordError, match="unknown reasoning_policy"):
        pr.admit_to_training_ready(make_record(), reasoning_policy="maybe")


# --------------------------------------------------------------------------
# condition 3: privacy and review
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["UNKNOWN", "NOT_SCANNED", "FAIL", "QUARANTINED", ""])
def test_unpassed_privacy_blocks(status: str) -> None:
    assert not pr.admit_to_training_ready(make_record(privacy_status=status)).admitted


def test_unreviewed_records_block() -> None:
    assert not pr.admit_to_training_ready(make_record(quality_review="NOT_REVIEWED")).admitted


def test_final_row_automated_redaction_is_an_admissible_privacy_result() -> None:
    decision = pr.admit_to_training_ready(
        make_record(privacy_status="PASS_AUTOMATED_REDACTION")
    )
    assert decision.admitted, decision.reasons


def test_review_vocabulary_distinguishes_automated_from_manual() -> None:
    """A pack must never imply a human read a record when none did."""
    assert "AUTOMATED_REVIEW_PASS" in pr.ADMISSIBLE_QUALITY_REVIEW
    assert "AUTOMATED_REVIEW_PASS" not in pr.MANUAL_REVIEW_VALUES
    assert "MANUAL_REVIEW_PASS" in pr.MANUAL_REVIEW_VALUES


# --------------------------------------------------------------------------
# condition 4: leakage group
# --------------------------------------------------------------------------


def test_a_group_shared_with_heldout_blocks() -> None:
    decision = pr.admit_to_training_ready(make_record(), heldout_group_ids={"GRP-1"})
    assert not decision.admitted
    assert decision.checks["leakage"] == "BLOCKED"


def test_any_secondary_group_shared_with_heldout_blocks() -> None:
    record = make_record(split_group_ids=["GRP-1", "SESSION-PRIVATE"])
    decision = pr.admit_to_training_ready(
        record, heldout_group_ids={"SESSION-PRIVATE"}
    )
    assert not decision.admitted
    assert decision.checks["leakage"] == "BLOCKED"


def test_heldout_records_are_never_training_inputs() -> None:
    decision = pr.admit_to_training_ready(make_record(split=pr.HELDOUT))
    assert not decision.admitted
    assert decision.checks["split"] == "HELD_OUT"


# --------------------------------------------------------------------------
# condition 5: source status
# --------------------------------------------------------------------------


def test_quarantined_source_ingot_blocks() -> None:
    record = make_record(source={"source_ingot_status": "QUARANTINED"})
    assert not pr.admit_to_training_ready(record).admitted


@pytest.mark.parametrize("status", ["HOLD", "FAIL", "ERROR"])
def test_non_pass_foundry_status_blocks(status: str) -> None:
    record = make_record(source={"source_foundry_status": status})
    assert not pr.admit_to_training_ready(record).admitted


def test_v2_still_requires_passed_and_pass() -> None:
    record = make_record(
        schema_id=pr.V2_SCHEMA_ID,
        source={"source_ingot_status": "RIGHTS_ASSERTED", "source_foundry_status": "PASS"},
    )
    assert pr.validate_record(record)
    assert not pr.admit_to_training_ready(record).admitted


def test_v3_admits_rights_asserted_plus_foundry_pass() -> None:
    record = make_record(
        schema_id=pr.V3_SCHEMA_ID,
        source={"source_ingot_status": "RIGHTS_ASSERTED", "source_foundry_status": "PASS"},
    )
    assert pr.validate_record(record) == []
    assert pr.admit_to_training_ready(record).admitted


def test_v3_admits_passed_plus_foundry_pass() -> None:
    record = make_record(schema_id=pr.V3_SCHEMA_ID)
    assert pr.validate_record(record) == []
    assert pr.admit_to_training_ready(record).admitted


def test_v3_rejects_rights_asserted_without_foundry() -> None:
    record = make_record(
        schema_id=pr.V3_SCHEMA_ID,
        source={"source_ingot_status": "RIGHTS_ASSERTED", "source_foundry_status": None},
    )
    assert pr.validate_record(record)
    assert not pr.admit_to_training_ready(record).admitted


def test_v3_rejects_hold_plus_foundry_pass() -> None:
    record = make_record(
        schema_id=pr.V3_SCHEMA_ID,
        source={"source_ingot_status": "HOLD", "source_foundry_status": "PASS"},
    )
    assert pr.validate_record(record)
    assert not pr.admit_to_training_ready(record).admitted


def test_v3_does_not_translate_rights_asserted() -> None:
    record = make_record(
        schema_id=pr.V3_SCHEMA_ID,
        source={"source_ingot_status": "RIGHTS_ASSERTED", "source_foundry_status": "PASS"},
    )
    assert record["source"]["source_ingot_status"] == "RIGHTS_ASSERTED"
    assert pr.admit_to_training_ready(record).admitted
    assert record["source"]["source_ingot_status"] == "RIGHTS_ASSERTED"


def test_v3_rejects_assertion_body() -> None:
    record = make_record(schema_id=pr.V3_SCHEMA_ID)
    record["rights_assertion"] = _rights_assertion()
    errors = pr.validate_record(record)
    assert errors
    assert any("must not carry a rights assertion body" in item for item in errors)


def test_v3_requires_hash_only_binding() -> None:
    record = make_record(schema_id=pr.V3_SCHEMA_ID)
    assert record["rights_assertion"] is None
    assert record["rights_assertion_binding"]["disposition"] == pr.ASSERTION_DISPOSITION_EXTERNAL
    assert pr.validate_record(record) == []


# --------------------------------------------------------------------------
# partitioning a set
# --------------------------------------------------------------------------


def test_partition_derives_heldout_groups_from_the_records_themselves() -> None:
    """A caller cannot forget to pass the leakage groups in."""
    records = [
        make_record(record_id="T-1", split=pr.TRAIN, split_group_id="SHARED"),
        make_record(record_id="E-1", split=pr.HELDOUT, split_group_id="SHARED"),
        make_record(record_id="T-2", split=pr.TRAIN, split_group_id="CLEAN"),
    ]
    result = pr.partition_records(records)
    assert result["heldout_group_ids"] == ["SHARED"]
    assert result["counts"] == {"total": 3, "admitted": 1, "rejected": 2}
    assert [r["record_id"] for r in result["admitted"]] == ["T-2"]


def test_partition_derives_every_heldout_lineage_key_itself() -> None:
    records = [
        make_record(
            record_id="T-1",
            split_group_id="TRAIN-PRIMARY",
            split_group_ids=["TRAIN-PRIMARY", "SHARED-SESSION"],
        ),
        make_record(
            record_id="E-1",
            split=pr.HELDOUT,
            split_group_id="EVAL-PRIMARY",
            split_group_ids=["EVAL-PRIMARY", "SHARED-SESSION", "SOURCE-FAMILY"],
        ),
        make_record(record_id="T-2", split_group_id="CLEAN"),
    ]
    result = pr.partition_records(records)
    assert result["heldout_group_ids"] == [
        "EVAL-PRIMARY",
        "SHARED-SESSION",
        "SOURCE-FAMILY",
    ]
    assert [record["record_id"] for record in result["admitted"]] == ["T-2"]
    assert result["counts"] == {"total": 3, "admitted": 1, "rejected": 2}


def test_partition_reports_why_records_were_rejected() -> None:
    records = [
        make_record(record_id="A", rights_status=pr.RIGHTS_UNKNOWN),
        make_record(record_id="B", rights_status=pr.RIGHTS_UNKNOWN),
        make_record(record_id="C", authorship_class=pr.MIXED),
    ]
    result = pr.partition_records(records)
    assert result["counts"]["admitted"] == 0
    assert sum(result["rejection_reasons"].values()) >= 3


def test_partition_records_the_active_reasoning_policy() -> None:
    result = pr.partition_records([make_record()], reasoning_policy=pr.REASONING_ALLOW)
    assert result["reasoning_policy"] == pr.REASONING_ALLOW


def test_a_malformed_record_is_rejected_not_admitted() -> None:
    result = pr.partition_records([{"record_id": "junk"}])
    assert result["counts"]["admitted"] == 0
    assert any("not valid" in r for d in result["decisions"] for r in d["reasons"])
