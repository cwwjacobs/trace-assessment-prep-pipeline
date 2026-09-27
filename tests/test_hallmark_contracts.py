"""Hallmark decides whether a product passes, so it has to be installable.

These tests exist because it was not. ``goldtrace_hallmark`` lived under
``donor_components/`` where ``packages.find`` could not see it, ten of thirteen
contracts never shipped as package data, and the record-map contract the
verifier loads was absent from the repository entirely. A Production mint had
no installed verifier to call, so hallmark reports were hand-written instead —
which is how a lot with an empty dataset came to carry a PASS.

Each test below pins one link in that chain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jsonschema import Draft202012Validator

import goldtrace_hallmark
from goldtrace_hallmark.verify import _contract
from goldtrace_mint.build import RECORD_MAP_SCHEMA_ID, _record_map_row
from goldtrace_refinery.paths import find_contract

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_CONTRACTS = COMPONENT_ROOT / "contracts"
SHIPPED_CONTRACTS = COMPONENT_ROOT / "src" / "goldtrace_refinery" / "contracts"

RECORD_MAP_CONTRACT = "mint-record-map.v1.schema.json"
SHA256 = "a" * 64


def test_hallmark_is_importable_from_the_packaged_location() -> None:
    """A verifier under donor_components cannot be found by packages.find."""
    module_path = Path(goldtrace_hallmark.__file__).resolve()
    assert "donor_components" not in module_path.parts, (
        f"goldtrace_hallmark must live under src/ to ship in the wheel; "
        f"found at {module_path}"
    )
    assert module_path.parents[1].name == "src"


def test_every_repository_contract_ships_as_package_data() -> None:
    """Repository-relative lookup does not exist once the package is installed."""
    repository = {path.name for path in REPOSITORY_CONTRACTS.iterdir() if path.is_file()}
    shipped = {path.name for path in SHIPPED_CONTRACTS.iterdir() if path.is_file()}
    assert repository, "the repository contract directory is empty"
    assert repository - shipped == set(), (
        f"contracts missing from package data (an installed release cannot "
        f"resolve these): {sorted(repository - shipped)}"
    )


def test_shipped_contracts_have_not_drifted_from_the_repository() -> None:
    """Two copies exist, so prove they agree rather than assuming it."""
    drifted = [
        name.name
        for name in REPOSITORY_CONTRACTS.iterdir()
        if name.is_file()
        and (SHIPPED_CONTRACTS / name.name).is_file()
        and (SHIPPED_CONTRACTS / name.name).read_bytes() != name.read_bytes()
    ]
    assert not drifted, f"repository and shipped contracts disagree: {drifted}"


def test_hallmark_resolves_contracts_through_the_shared_resolver() -> None:
    """Walking parent directories for contracts/ only works in a source tree."""
    report_contract = _contract("hallmark-report.v1.schema.json")
    assert report_contract["$id"] == "goldtrace.hallmark.report.v1"


def test_the_record_map_contract_is_resolvable() -> None:
    """The exact absence that made the verifier crash on any non-empty lot.

    ``verify_product_lot`` validates each record-map entry inside its iteration
    over ``provenance/record-map.jsonl``. While this contract was missing the
    verifier could only complete on lots whose record map was empty.
    """
    assert find_contract(RECORD_MAP_CONTRACT).is_file()
    assert _contract(RECORD_MAP_CONTRACT)["$id"] == RECORD_MAP_SCHEMA_ID


def _record_map_row_for(recipe_id: str) -> dict[str, object]:
    provenance = {
        "session_id": "sess-1",
        "turn_id": "turn-1",
        "user_input_event_ids": ["evt-user"],
        "agent_message_event_ids": ["evt-agent"],
        "request_event_ids": ["evt-request"],
        "result_event_ids": ["evt-result"],
        "effect_event_ids": ["evt-effect"],
        "source_interactive_evidence_sha256": SHA256,
        "source_index_sha256": SHA256,
        "normalized_tool_calls": [
            {
                "normalized_call_id": "call-1",
                "native_id_field": "id",
                "native_call_id": "native-1",
                "request_event_id": "evt-request",
                "result_event_id": "evt-result",
                "effect_event_ids": ["evt-effect"],
                "result_state": "RESULT_OBSERVED",
            }
        ],
    }
    return _record_map_row(
        content={"_provenance": provenance, "extractor_note": "kept out of the row"},
        model_row={"messages": [{"role": "user", "content": "hi"}]},
        recipe_id=recipe_id,
        record_id="rec-1",
        split="train",
        row_index=0,
        ingot={"source_bundle_hash": SHA256, "mine_run_id": "run-1"},
        decision={
            "ingot_id": "ing-1",
            "ingot_hash": SHA256,
            "split_group_id": "grp-1",
            "decision_id": "dec-1",
        },
    )


@pytest.mark.parametrize(
    "recipe_id",
    ["sft-chat-v1", "tool-use-v1", "dpo-preference-v1", "recovery-v1", "eval-pairwise-v1"],
)
def test_producer_rows_satisfy_the_contract(recipe_id: str) -> None:
    """The contract is derived from the producer, so every recipe must validate."""
    row = _record_map_row_for(recipe_id)
    errors = list(Draft202012Validator(_contract(RECORD_MAP_CONTRACT)).iter_errors(row))
    assert not errors, f"{recipe_id}: {[error.message for error in errors]}"


def test_producer_rows_validate_when_provenance_is_absent() -> None:
    """Every provenance field is nullable; a bundle without one still mints."""
    row = _record_map_row(
        content={},
        model_row={"messages": []},
        recipe_id="sft-chat-v1",
        record_id="rec-2",
        split="validation",
        row_index=3,
        ingot={"source_bundle_hash": SHA256},
        decision={
            "ingot_id": "ing-2",
            "ingot_hash": SHA256,
            "split_group_id": "grp-2",
            "decision_id": "dec-2",
        },
    )
    assert row["dataset_path"] == "dataset/validation.jsonl"
    errors = list(Draft202012Validator(_contract(RECORD_MAP_CONTRACT)).iter_errors(row))
    assert not errors, [error.message for error in errors]


@pytest.mark.parametrize(
    ("field", "value", "why"),
    [
        ("rendered_row_sha256", "not-a-digest", "row hash must be a sha256"),
        ("split", "holdout", "split is train or validation only"),
        ("dataset_path", "dataset/holdout.jsonl", "dataset path is bound to the split"),
        ("row_index", -1, "a row index cannot be negative"),
        ("source_ingot_hash", "", "an empty hash is not a binding"),
        ("record_id", "", "an unnamed record cannot be reconciled"),
        ("schema_id", "goldtrace.mint.record_map.v2", "the contract names its own version"),
    ],
)
def test_the_contract_rejects_a_malformed_row(field: str, value: object, why: str) -> None:
    """A contract that accepts everything proves nothing about what it accepted."""
    row = _record_map_row_for("sft-chat-v1")
    row[field] = value
    errors = list(Draft202012Validator(_contract(RECORD_MAP_CONTRACT)).iter_errors(row))
    assert errors, f"contract accepted a malformed {field}: {why}"


def test_the_contract_rejects_an_unknown_field() -> None:
    """Unbound fields would travel in the record map without being contracted."""
    row = _record_map_row_for("sft-chat-v1")
    row["smuggled_field"] = "not in the contract"
    errors = list(Draft202012Validator(_contract(RECORD_MAP_CONTRACT)).iter_errors(row))
    assert errors, "contract accepted a field it does not define"


def test_a_record_map_entry_round_trips_as_jsonl() -> None:
    """The verifier reads these back line by line; they must survive the trip."""
    row = _record_map_row_for("tool-use-v1")
    restored = json.loads(json.dumps(row))
    assert restored == row
    errors = list(
        Draft202012Validator(_contract(RECORD_MAP_CONTRACT)).iter_errors(restored)
    )
    assert not errors, [error.message for error in errors]
