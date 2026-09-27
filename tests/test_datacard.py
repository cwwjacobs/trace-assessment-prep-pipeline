"""Unit tests for Institutional DataCard generation and token statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from goldtrace_mint.build import build_product_lot
from goldtrace_mint.curators import ManualFileCurator
from goldtrace_mint.factory.datacard import (
    InstitutionalDataCard,
    TokenStatistics,
    compute_token_statistics,
    extract_task_taxonomy,
    generate_datacard,
    render_datacard_markdown,
)
from goldtrace_refinery.cast import cast_mine_bundle


def test_token_statistics_computation():
    records = [
        {
            "record_id": "rec-1",
            "prompt_tokens": 120,
            "cot_reasoning_tokens": 450,
            "tool_execution_turns": 4,
        },
        {
            "record_id": "rec-2",
            "prompt_tokens": 80,
            "cot_reasoning_tokens": 150,
            "tool_execution_turns": 2,
        },
        {
            "record_id": "rec-3",
            "messages": [
                {"role": "system", "content": "You are a specialized agent."},
                {"role": "user", "content": "Run analysis on logs."},
                {"role": "assistant", "reasoning": "First let's check log files in the directory.", "content": "Checking logs."},
                {"role": "tool_call", "content": "ls logs"},
                {"role": "tool", "content": "log1.txt log2.txt"},
                {"role": "assistant", "content": "<think>Now parsing log entries.</think>Found 0 errors."},
            ],
        },
    ]

    stats = compute_token_statistics(records)

    assert stats.total_records == 3
    assert stats.total_tokens > 0
    assert stats.total_prompt_tokens >= 200
    assert stats.total_cot_tokens >= 600
    assert stats.mean_prompt_length > 0
    assert stats.mean_cot_reasoning_tokens > 0
    assert stats.mean_tool_execution_turns >= 2.0
    assert stats.min_prompt_length > 0
    assert stats.max_prompt_length >= 120


def test_task_taxonomy_extraction():
    product_spec = {
        "intended_use": ["agent_sft", "evaluation"],
        "required_coverage": ["lineage_present", "pii_redacted"],
        "target_dataset_form": {"modality": "agent_trace"},
    }
    records = [
        {
            "scenario_id": "owned-network-investigation",
            "labels": {"reason_codes": ["clean_trace", "high_coherence"]},
        }
    ]

    taxonomy = extract_task_taxonomy(product_spec, records)
    assert "agent_sft" in taxonomy
    assert "evaluation" in taxonomy
    assert "coverage:lineage_present" in taxonomy
    assert "coverage:pii_redacted" in taxonomy
    assert "modality:agent_trace" in taxonomy
    assert "scenario:owned-network-investigation" in taxonomy
    assert "tag:clean_trace" in taxonomy


#: Point these at a sealed Labyrinth bundle and a mint rubric to run the
#: end-to-end DataCard check. Previously this test read a fixed path under one
#: operator's home directory, so it passed on exactly one machine and would have
#: failed everywhere else.
BUNDLE_ENV_VAR = "GOLDTRACE_DATACARD_E2E_BUNDLE"
RUBRIC_ENV_VAR = "GOLDTRACE_DATACARD_E2E_RUBRIC"


def test_generate_datacard_e2e(tmp_path: Path, monkeypatch):
    import os

    bundle_setting = os.environ.get(BUNDLE_ENV_VAR)
    rubric_setting = os.environ.get(RUBRIC_ENV_VAR)
    if not bundle_setting or not rubric_setting:
        pytest.skip(
            f"set {BUNDLE_ENV_VAR} to a sealed Labyrinth bundle and {RUBRIC_ENV_VAR} "
            f"to a mint rubric JSON to run the end-to-end DataCard check"
        )

    sealed_bundle = Path(bundle_setting).expanduser().resolve()
    rubric_path = Path(rubric_setting).expanduser().resolve()
    assert sealed_bundle.is_dir(), f"{BUNDLE_ENV_VAR} is not a directory: {sealed_bundle}"
    assert rubric_path.is_file(), f"{RUBRIC_ENV_VAR} is not a file: {rubric_path}"

    cast_dir = tmp_path / "cast"
    cast_dir.mkdir()
    cast_result = cast_mine_bundle(sealed_bundle, cast_dir)
    ingot_path = Path(cast_result["ingot_path"])
    ingot = json.loads(ingot_path.read_text(encoding="utf-8"))

    # Curation decisions
    decisions_path = tmp_path / "curation_decisions.json"
    decisions_path.write_text(
        json.dumps(
            [
                {
                    "ingot_id": ingot["ingot_id"],
                    "decision": "INCLUDE",
                    "reason_codes": ["fixture_include", "lineage_present"],
                    "proposed_dataset_role": "train",
                    "split_group_id": ingot["ingot_id"],
                    "fixture_only": True,
                    "confidence": 1.0,
                    "reviewer_identity": "fixture_manual_file",
                }
            ]
        ),
        encoding="utf-8",
    )

    curator = ManualFileCurator(decisions_path)
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    product_spec = {
        "schema_id": "goldtrace.mint.product_spec.v1",
        "product_id": "datacard-fixture-product",
        "product_version": "1.0.0",
        "fixture_only": True,
        "intended_use": ["institutional_datacard_verification"],
        "prohibited_use": ["uncertified_sale"],
        "target_dataset_form": {"record_format": "jsonl", "modality": "agent_trace"},
        "required_coverage": ["lineage_present"],
        "quality_rubric": {"rubric_id": "fixture_rubric", "rubric_version": "1.0.0"},
        "inclusion_requirements": ["lineage_present"],
        "exclusion_requirements": ["unresolved_privacy_high_severity"],
        "privacy_requirements": {"redaction": "required"},
        "rights_requirements": {"license": "CC-BY-4.0-COMMERCIAL"},
        "split_policy": {"leakage_group_field": "split_group_id", "train_ratio": 1.0, "validation_ratio": 0.0, "eval_held_out": True},
        "eval_requirements": {"grader": "exact_label_match_v1"},
        "hallmark_requirements": {"mode": "fixture"},
        "release_restrictions": {"fixture_only": True},
    }

    lot_dir = tmp_path / "product_lot"
    lot_result = build_product_lot(
        ingot_paths=[ingot_path],
        product_spec=product_spec,
        rubric=rubric,
        curator=curator,
        out_dir=lot_dir,
    )

    card, json_path, md_path = generate_datacard(lot_dir)

    assert isinstance(card, InstitutionalDataCard)
    assert json_path.is_file()
    assert md_path.is_file()

    # Verify JSON content
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["schema_id"] == "goldtrace.mint.datacard.v1"
    assert data["product_id"] == "datacard-fixture-product"
    assert data["product_version"] == "1.0.0"
    assert data["license"] == "CC-BY-4.0-COMMERCIAL"
    assert data["lot_hash"] == lot_result["product_hash"]
    assert data["token_statistics"]["total_records"] > 0
    assert data["token_statistics"]["total_tokens"] > 0
    assert data["token_statistics"]["mean_prompt_length"] > 0
    assert "provenance_chain" in data
    assert data["provenance_chain"]["source_ingots_count"] == 1

    assert data["provenance_chain"]["product_hash"] == lot_result["product_hash"]
    assert "gebru_sections" in data
    assert data["gebru_sections"]["motivation"]["purpose"] != ""
    assert "Refinery Invariant Verification" in md_path.read_text(encoding="utf-8")
    assert "Token & Interaction Statistics" in md_path.read_text(encoding="utf-8")
