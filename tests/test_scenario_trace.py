from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from goldtrace_mint.content import extract_content
from goldtrace_refinery.cast import cast_mine_bundle
from goldtrace_refinery.foundry_receipt import FoundryReceiptError
from goldtrace_refinery.ingot_schema import validate_ingot
from goldtrace_refinery.scenario_trace import verify_scenario_trace_evidence

try:
    from evalfoundry.handoff import ReceiptIssuer, evaluate_bundle, write_receipt
    from goldentrace.scenario_trace_capture import capture_scenario_trace

    HAS_CHAIN = True
except ImportError:  # pragma: no cover - standalone component checkout
    HAS_CHAIN = False

requires_chain = pytest.mark.skipif(
    not HAS_CHAIN,
    reason="sibling Labyrinth/Foundry packages are not importable",
)


def _repo_sibling(name: str) -> Path | None:
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor.parent / name
        if candidate.is_dir():
            return candidate
    return None


def _proof_fixture() -> Path | None:
    labyrinth = _repo_sibling("1.GTDataworks-Labyrinth")
    if labyrinth is None:
        return None
    candidate = labyrinth / "tests" / "fixtures" / "scenario-trace-proof-v0.1"
    return candidate if (candidate / "manifest.json").is_file() else None


def _pack_path() -> Path | None:
    foundry = _repo_sibling("2.GTDataworks-Foundry")
    if foundry is None:
        return None
    candidate = (
        foundry
        / "evalfoundry"
        / "admission_packs"
        / "scenario-trace-admission-v1"
        / "pack.json"
    )
    return candidate if candidate.is_file() else None


def _prepare_chain(tmp_path: Path):
    proof = _proof_fixture()
    pack = _pack_path()
    if proof is None or pack is None:
        pytest.skip("sibling causal proof or Foundry admission pack is unavailable")
    capture = capture_scenario_trace(proof, runs_root=tmp_path / "runs")
    key = b"scenario-trace-test-key-material!!"[:32]
    assert len(key) == 32
    issuer = ReceiptIssuer(
        issuer_id="scenario-trace-test-foundry",
        key_id="scenario-trace-test-key-v1",
        authentication_key=key,
    )
    receipt = evaluate_bundle(capture.bundle_path, pack, receipt_issuer=issuer)
    assert receipt["status"] == "PASS"
    receipt_path = write_receipt(receipt, tmp_path / "foundry-receipt.json")
    key_path = tmp_path / "foundry-key.bin"
    key_path.write_bytes(key)
    os.chmod(key_path, 0o600)
    trust_path = tmp_path / "trust-policy.json"
    trust_path.write_text(
        json.dumps(
            {
                "schema_id": "goldtrace.refinery.foundry_trust_policy.v1",
                "issuer_id": issuer.issuer_id,
                "authentication_method": "hmac-sha256",
                "key_id": issuer.key_id,
                "authentication_key_file": str(key_path),
                "evaluation_packs": [
                    {
                        "pack_id": receipt["evaluation_pack"]["pack_id"],
                        "pack_version": receipt["evaluation_pack"]["pack_version"],
                        "pack_hash": receipt["evaluation_pack"]["pack_hash"],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(trust_path, 0o600)
    return proof, capture, receipt_path, trust_path


@requires_chain
@pytest.mark.fixture_e2e
def test_scenario_trace_three_tier_cast_produces_control_reference_ingot(
    tmp_path: Path,
) -> None:
    proof, capture, receipt, trust = _prepare_chain(tmp_path)
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in proof.iterdir()
        if path.is_file()
    }
    out = cast_mine_bundle(
        capture.bundle_path,
        tmp_path / "ingot",
        foundry_receipt=receipt,
        foundry_trust_policy=trust,
        require_foundry_receipt=True,
    )
    assert out["refinery_status"] == "PASSED"
    assert out["source_family"] == "CONTROL_POLICY"
    assert out["processor"] == "GTDataworks-Refinery"
    assert out["output_class"] == "CONTROL_REFERENCE_INGOT"
    ingot_root = Path(out["ingot_path"]).parent
    ingot = json.loads(Path(out["ingot_path"]).read_text(encoding="utf-8"))
    assert not validate_ingot(ingot)
    assert ingot["mechanical_run_status"] == "COMPLETED"
    assert ingot["scenario_id"] == "scenario.dynamic-resource.v1"
    evidence = json.loads(
        (ingot_root / "scenario-trace-evidence.json").read_text(encoding="utf-8")
    )
    verify_scenario_trace_evidence(evidence, ingot=ingot)
    route = json.loads(
        (ingot_root / "processor-routing-receipt.json").read_text(encoding="utf-8")
    )
    assert route["existing_canon_modified"] is False
    assert route["crucible_used"] is False
    assert route["selected_route"]["gold_eligible"] is False
    records = extract_content(
        ingot,
        "recovery-v1",
        capture.bundle_path.parent,
        ingot_root=ingot_root,
    )
    assert len(records) == 1
    assert [item["trajectory_label"] for item in records[0]["attempts"]] == [
        "failure",
        "correction",
        "recovery",
        "none",
    ]
    rendered = json.dumps(records, sort_keys=True)
    for forbidden in (
        "world_state",
        "hidden_event_refs",
        "state_delta",
        "event.resource-a-unavailable",
        "/resources/A/available",
    ):
        assert forbidden not in rendered
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in proof.iterdir()
        if path.is_file()
    }
    assert after == before


@requires_chain
def test_scenario_trace_cast_requires_authenticated_foundry_before_writes(
    tmp_path: Path,
) -> None:
    _, capture, _, _ = _prepare_chain(tmp_path)
    out_dir = tmp_path / "missing-foundry"
    with pytest.raises(FoundryReceiptError, match="requires the authenticated three-tier"):
        cast_mine_bundle(
            capture.bundle_path,
            out_dir,
            require_foundry_receipt=False,
        )
    assert not out_dir.exists()


@requires_chain
@pytest.mark.fixture_e2e
def test_scenario_trace_derivative_tamper_blocks_mint_extraction(tmp_path: Path) -> None:
    _, capture, receipt, trust = _prepare_chain(tmp_path)
    out = cast_mine_bundle(
        capture.bundle_path,
        tmp_path / "ingot",
        foundry_receipt=receipt,
        foundry_trust_policy=trust,
        require_foundry_receipt=True,
    )
    ingot_root = Path(out["ingot_path"]).parent
    ingot = json.loads(Path(out["ingot_path"]).read_text(encoding="utf-8"))
    evidence_path = ingot_root / "scenario-trace-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["trajectory"]["steps"][0]["observation_given_to_model"]["hidden"] = {
        "event_fired": True
    }
    evidence_path.write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        extract_content(
            ingot,
            "recovery-v1",
            capture.bundle_path.parent,
            ingot_root=ingot_root,
        )


def test_scenario_trace_contract_copies_are_synced() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "refinery-scenario-trace-evidence.v1.schema.json",
        "processor-routing-receipt.v1.schema.json",
    ):
        assert (root / "contracts" / name).read_bytes() == (
            root / "src" / "goldtrace_refinery" / "contracts" / name
        ).read_bytes()
