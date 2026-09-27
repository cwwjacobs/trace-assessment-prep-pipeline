from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from goldtrace_refinery.hashing import sha256_json
from goldtrace_refinery.ingot_mold import (
    MoldError,
    _core_hash_body,
    build_domain_ingot_core,
    cast_files,
    load_mold,
    verify_domain_ingot_core,
)


def _source_ingot() -> dict:
    return {
        "schema_id": "goldtrace.refinery.ingot.v1",
        "ingot_id": "GTI-sentinel-unit",
        "source_class": "goldtrace_mine_bundle",
        "source_bundle_hash": "a" * 64,
        "source_manifest_hash": "b" * 64,
        "evaluation_path": "three_tier_labyrinth_foundry_refinery",
        "foundry_evaluation": {
            "receipt_schema_id": "gtdataworks.foundry.evaluation_receipt.v1",
            "receipt_schema_version": "1.0.0",
            "receipt_hash": "c" * 64,
            "status": "PASS",
            "evaluation_pack_id": "sentinel-cyber-eval",
            "evaluation_pack_version": "1.0.0",
            "evaluation_pack_hash": "d" * 64,
            "foundry_code_sha256": "e" * 64,
            "deterministic": True,
            "evidence_gaps": [],
            "evaluated_at": "2026-08-13T10:00:00Z",
            "trust": {
                "channel": "hmac-sha256",
                "issuer_id": "foundry-operator",
                "key_id": "foundry-key-1",
                "pack_identity_verified": True,
                "evaluation_consistency_verified": True,
            },
        },
        "mine_run_id": "run-sentinel-1",
        "scenario_id": "defensive-boundary-confusion",
        "scenario_version": "1.0.0",
        "mechanical_run_status": "COMPLETED",
        "seal_status": "SEALED",
        "evidence_gap_ids": [],
        "normalized_event_stream_hash": "f" * 64,
        "deterministic_results": {},
        "redaction": {"unresolved_high_severity": 0},
        "quarantine": {"status": "CLEAR"},
        "dedup": {"exact_id": "exact-1", "structural_id": "structural-1"},
        "lineage_parents": [],
        "refinery_receipt_hash": "1" * 64,
        "refinery_status": "PASSED",
        "created_at": "2026-08-13T10:01:00Z",
    }


def _record() -> dict:
    return {
        "schema_id": "sentinel-sanctuary.cyber-record.v1",
        "emission_type": "PROPOSAL_DECISION",
        "case_type": "CONFIRMED_INCIDENT",
        "scenario_family_id": "boundary-confusion",
        "fixture_root_sha256": "2" * 64,
        "root_cause_mechanic_id": "authority-vs-observation",
        "narrative_id": "narrative-a",
        "replay_mode": "CAPTURED_EXACT",
        "lineage": {
            "parent_run_sha256": "3" * 64,
            "checkpoint_sha256": "4" * 64,
            "compared_core_sha256s": ["6" * 64, "5" * 64],
        },
        "governance": {
            "authorization_receipt_sha256": "7" * 64,
            "rights_manifest_sha256": "8" * 64,
            "privacy_ruleset_sha256": "9" * 64,
            "retention_policy_sha256": "a" * 64,
            "provider_disclosure_policy_sha256": "b" * 64,
            "commercial_rights_status": "VERIFIED",
            "provider_submission_authorized": True,
            "raw_evidence_disposition": "ENCRYPTED_QUARANTINE_ONLY",
        },
        "evaluation": {
            "oracle_sha256": "c" * 64,
            "scoring_rules_sha256": "d" * 64,
            "split_policy_sha256": "e" * 64,
            "split_assignment_sha256": "f" * 64,
            "split_role": "HOLDOUT",
            "contamination_scan_sha256": "0" * 64,
            "contamination_status": "CLEAR",
            "evaluation_case_sha256s": ["2" * 64, "1" * 64],
            "evaluation_result_sha256s": ["4" * 64, "3" * 64],
        },
        "model_witness": {
            "api_family": "openai.responses.v1",
            "endpoint_origin_sha256": "5" * 64,
            "requested_model_id": "gpt-snapshot-test",
            "returned_model_id": "gpt-snapshot-test",
            "model_snapshot_pinned": True,
            "system_fingerprint_sha256": "6" * 64,
            "request_sha256": "7" * 64,
            "response_sha256": "8" * 64,
            "request_parameters_sha256": "9" * 64,
            "tool_schema_sha256": "a" * 64,
            "provider_request_id_sha256": "b" * 64,
            "client_request_id_sha256": "c" * 64,
            "usage_sha256": "d" * 64,
            "data_control_profile_sha256": "e" * 64,
            "store_requested": False,
            "seed_requested": 17,
            "seed_support": "SUPPORTED",
            "output_reproducibility": "NOT_CLAIMED",
        },
        "execution_witness": {
            "executor": "cinderfield",
            "profile": "agent-probe/v0",
            "assurance_class": "SUPPORTED_HOST_VERIFIED",
            "execution_envelope_sha256": "f" * 64,
            "adapter_schema_sha256": "0" * 64,
            "observer_issuer_id": "cinderfield-observer",
            "observer_key_id": "observer-key-1",
            "observer_authentication": "ASYMMETRIC_VERIFIED",
            "road_frozen_sha256": "1" * 64,
            "road_walked_sha256": "2" * 64,
            "road_diff_sha256": "3" * 64,
            "execution_receipt_sha256": "4" * 64,
            "runtime_identity_sha256": "5" * 64,
            "runtime_asset_lock_sha256": "6" * 64,
            "job_image_sha256": "7" * 64,
            "host_preflight_sha256": "8" * 64,
            "proxy_peer_sha256": "9" * 64,
            "guest_evidence_root": "a" * 64,
            "ciphertext_sha256": "b" * 64,
            "ciphertext_size_bytes": 4096,
            "egress_event_root": "c" * 64,
            "host_lifecycle_root": "d" * 64,
            "evidence_recoverability": "DECRYPTABLE_VERIFIED",
            "gate_decision": "DENY",
            "cleanup_status": "VERIFIED",
            "credential_channel_closed": True,
            "provider_credential_revocation_receipt_sha256": None,
        },
        "finding_codes": ["BOUNDARY_BREACH", "AUTHORITY_CONFUSION"],
        "evidence_ref_sha256s": ["f" * 64, "e" * 64],
        "decision_episode_sha256s": ["d" * 64],
        "gap_codes": [],
        "split_group_ids": [
            "source:fixture-owned",
            "narrative:narrative-a",
            "mechanic:authority-vs-observation",
            "fixture:fixture-root-a",
            "family:boundary-confusion",
            "checkpoint:checkpoint-a",
        ],
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cast(
    tmp_path: Path,
    ingot: dict | None = None,
    record: dict | None = None,
    name: str = "out",
):
    ingot_path = tmp_path / f"{name}-ingot.json"
    record_path = tmp_path / f"{name}-record.json"
    _write_json(ingot_path, ingot or _source_ingot())
    _write_json(record_path, record or _record())
    result = cast_files(ingot_path, record_path, tmp_path / name)
    return result, tmp_path / name


def test_new_contracts_are_valid_and_packaged_without_drift() -> None:
    root = Path(__file__).resolve().parents[1]
    names = (
        "domain-ingot-core.v1.schema.json",
        "domain-ingot-cast-request.v1.schema.json",
        "ingot-mold-module.v1.schema.json",
        "sentinel-sanctuary.cyber-record.v1.schema.json",
    )
    for name in names:
        canonical = root / "contracts" / name
        packaged = root / "src" / "goldtrace_refinery" / "contracts" / name
        Draft202012Validator.check_schema(json.loads(canonical.read_text(encoding="utf-8")))
        assert packaged.read_bytes() == canonical.read_bytes()
    assert (
        root / "src" / "goldtrace_refinery" / "contracts" / "sentinel-sanctuary.cyber.v1.json"
    ).read_bytes() == (root / "contracts" / "sentinel-sanctuary.cyber.v1.json").read_bytes()
    load_mold()


def test_semantic_core_is_stable_across_wrapper_time_hashes_and_set_order() -> None:
    first_ingot = _source_ingot()
    second_ingot = copy.deepcopy(first_ingot)
    second_ingot["created_at"] = "2030-01-01T00:00:00Z"
    second_ingot["refinery_receipt_hash"] = "2" * 64
    second_ingot["foundry_evaluation"]["evaluated_at"] = "2030-01-01T00:00:01Z"
    second_ingot["foundry_evaluation"]["receipt_hash"] = "3" * 64

    first_record = _record()
    second_record = copy.deepcopy(first_record)
    for key in ("finding_codes", "evidence_ref_sha256s", "split_group_ids"):
        second_record[key].reverse()
    second_record["lineage"]["compared_core_sha256s"].reverse()
    second_record["evaluation"]["evaluation_case_sha256s"].reverse()

    first = build_domain_ingot_core(first_ingot, first_record)
    second = build_domain_ingot_core(second_ingot, second_record)
    assert first == second
    assert first["claims"]["release_posture"] == "HOLD"
    assert first["claims"]["provider_disclosure_laundered"] is False


def test_exact_cast_request_changes_while_semantic_core_remains_stable(tmp_path: Path) -> None:
    first_result, first_out = _cast(tmp_path, name="first")
    second_ingot = _source_ingot()
    second_ingot["created_at"] = "2030-01-01T00:00:00Z"
    second_ingot["refinery_receipt_hash"] = "2" * 64
    second_ingot["foundry_evaluation"]["receipt_hash"] = "3" * 64
    second_result, second_out = _cast(tmp_path, ingot=second_ingot, name="second")

    assert first_result["core_sha256"] == second_result["core_sha256"]
    assert (first_out / "ingot-core.json").read_bytes() == (
        second_out / "ingot-core.json"
    ).read_bytes()
    assert first_result["request_sha256"] != second_result["request_sha256"]
    assert (first_out / "cast-request.json").read_bytes() != (
        second_out / "cast-request.json"
    ).read_bytes()


def test_cast_is_idempotent_private_and_baseline_gated(tmp_path: Path) -> None:
    result, output = _cast(tmp_path)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    ingot_path = tmp_path / "out-ingot.json"
    record_path = tmp_path / "out-record.json"
    repeated = cast_files(ingot_path, record_path, output)
    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert repeated == result
    assert before == after
    assert set(before) == {"ingot-core.json", "cast-request.json"}
    assert all((path.stat().st_mode & 0o077) == 0 for path in output.iterdir())

    core = json.loads((output / "ingot-core.json").read_text(encoding="utf-8"))
    request = json.loads((output / "cast-request.json").read_text(encoding="utf-8"))
    assert request["issuance"] == {
        "authentication_status": "UNSIGNED",
        "hallmark_status": "NOT_RUN",
        "required_gates": [
            "ASYMMETRIC_FOUNDRY_ATTESTATION",
            "AUTHENTICATED_HALLMARK",
            "AUTHENTICATED_ISSUANCE",
            "HUMAN_SALE_RELEASE",
            "VAULT_OWNED_RECOMPUTATION",
        ],
        "sale_eligibility": "HOLD",
        "vault_status": "NOT_SUBMITTED",
    }
    forbidden_keys = {"prompt", "target", "response", "content", "created_at", "evaluated_at"}

    def keys(value):
        if isinstance(value, dict):
            yield from value
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert forbidden_keys.isdisjoint(set(keys(core)))


def test_uncertain_witnesses_become_explicit_gates(tmp_path: Path) -> None:
    record = _record()
    record["governance"]["commercial_rights_status"] = "HOLD"
    record["evaluation"]["contamination_status"] = "HOLD"
    record["gap_codes"] = ["MISSING_RUNTIME_PROOF"]
    record["model_witness"]["model_snapshot_pinned"] = False
    execution = record["execution_witness"]
    execution["assurance_class"] = "PORTABLE_VERIFIED_HARDWARE_PENDING"
    execution["observer_authentication"] = "LOCAL_HMAC_VERIFIED"
    execution["evidence_recoverability"] = "STRUCTURAL_ONLY"
    execution["gate_decision"] = "RESCOPE_REQUIRED"
    execution["cleanup_status"] = "UNVERIFIED"
    execution["credential_channel_closed"] = False
    _, output = _cast(tmp_path, record=record)
    request = json.loads((output / "cast-request.json").read_text(encoding="utf-8"))
    gates = set(request["issuance"]["required_gates"])
    assert {
        "ASYMMETRIC_EXECUTION_ATTESTATION",
        "COMMERCIAL_RIGHTS_VERIFICATION",
        "CONTAMINATION_CLEARANCE",
        "CREDENTIAL_CHANNEL_CLOSURE",
        "EVIDENCE_DECRYPTABILITY_PROOF",
        "EVIDENCE_GAPS_RESOLVED",
        "EXECUTION_SCOPE_RESOLUTION",
        "PINNED_MODEL_SNAPSHOT",
        "SUPPORTED_HOST_CINDERFIELD_PROOF",
        "VERIFIED_CLEANUP",
    } <= gates


@pytest.mark.parametrize(
    "mutate",
    [
        lambda ingot, record: ingot.update(
            evaluation_path="legacy_direct_labyrinth_to_refinery",
            foundry_evaluation=None,
        ),
        lambda ingot, record: ingot["quarantine"].update(status="HOLD"),
        lambda ingot, record: ingot["foundry_evaluation"].update(deterministic=False),
        lambda ingot, record: record["model_witness"].update(store_requested=True),
        lambda ingot, record: record["split_group_ids"].remove("family:boundary-confusion"),
        lambda ingot, record: record.update(raw_prompt="must never enter the core"),
    ],
)
def test_fail_closed_inputs_are_refused(mutate) -> None:
    ingot = _source_ingot()
    record = _record()
    mutate(ingot, record)
    with pytest.raises(MoldError):
        build_domain_ingot_core(ingot, record)


def test_tamper_and_non_normal_set_order_are_refused() -> None:
    mold = load_mold()
    core = build_domain_ingot_core(_source_ingot(), _record(), mold=mold)
    tampered = copy.deepcopy(core)
    tampered["record"]["finding_codes"][0] = "CHANGED_FINDING"
    with pytest.raises(MoldError):
        verify_domain_ingot_core(tampered, mold=mold)

    denormalized = copy.deepcopy(core)
    denormalized["record"]["evidence_ref_sha256s"].reverse()
    digest = sha256_json(_core_hash_body(denormalized))
    denormalized["core_sha256"] = digest
    denormalized["core_id"] = f"GTIC-{digest[:24]}"
    with pytest.raises(MoldError, match="normalized"):
        verify_domain_ingot_core(denormalized, mold=mold)


def test_symlink_input_and_mismatched_existing_output_are_refused(tmp_path: Path) -> None:
    ingot_path = tmp_path / "ingot.json"
    record_path = tmp_path / "record.json"
    _write_json(ingot_path, _source_ingot())
    _write_json(record_path, _record())
    linked = tmp_path / "linked-ingot.json"
    linked.symlink_to(ingot_path)
    with pytest.raises(MoldError, match="non-symlink"):
        cast_files(linked, record_path, tmp_path / "linked-out")

    cast_files(ingot_path, record_path, tmp_path / "out")
    (tmp_path / "out" / "ingot-core.json").write_bytes(b"{}\n")
    with pytest.raises(MoldError, match="bytes differ"):
        cast_files(ingot_path, record_path, tmp_path / "out")
