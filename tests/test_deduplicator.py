#!/usr/bin/env python3
"""
Unit and integration tests for SeedCurator and Deduplication Engine.
"""

import json
import pytest
from pathlib import Path

from goldtrace_mint.factory.deduplicator import (
    CurationSummary,
    DeduplicationRecord,
    SeedCurator,
    extract_seed_canonical_text,
)


class TestSeedCurator:
    @pytest.fixture
    def curator(self):
        return SeedCurator(minhash_threshold=0.85, ngram_size=13)

    def test_exact_duplicate_pruning(self, curator):
        raw_seeds = [
            {"seed_id": "seed-001", "task_prompt": "Implement a distributed ledger transaction verifier with Merkle trees."},
            {"seed_id": "seed-002", "task_prompt": "Implement a distributed ledger transaction verifier with Merkle trees."},  # Exact dup
            {"seed_id": "seed-003", "task_prompt": "Design a high throughput lock-free queue in C++20."},
        ]

        clean, summary, audit = curator.curate_seeds(raw_seeds)

        assert len(clean) == 2
        assert summary.total_ingested == 3
        assert summary.exact_duplicates_pruned == 1
        assert summary.final_clean_count == 2
        assert clean[0]["seed_id"] == "seed-001"
        assert clean[1]["seed_id"] == "seed-003"
        assert audit[1].status == "DUPLICATE_EXACT"

    def test_minhash_near_duplicate_pruning(self, curator):
        base_prompt = (
            "Write a robust Python parser for parsing complex nested network telemetry packets "
            "with full validation of header checksums, payload lengths, and custom protocol headers."
        )
        # Minor variation (near-duplicate)
        near_dup_prompt = (
            "Write a robust Python parser for parsing complex nested network telemetry packets "
            "with full validation of header checksums, payload lengths, and custom protocol headers today."
        )
        distinct_prompt = (
            "Develop an automated theorem prover algorithm based on resolution refutation and unification."
        )

        raw_seeds = [
            {"seed_id": "base-01", "task_prompt": base_prompt},
            {"seed_id": "near-02", "task_prompt": near_dup_prompt},
            {"seed_id": "distinct-03", "task_prompt": distinct_prompt},
        ]

        clean, summary, audit = curator.curate_seeds(raw_seeds)

        assert len(clean) == 2
        assert summary.minhash_duplicates_pruned == 1
        assert audit[1].status == "DUPLICATE_MINHASH"
        assert audit[1].jaccard_similarity >= 0.80

    def test_benchmark_contamination_pruning(self, curator):
        humaneval_prompt = (
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n"
            "    given threshold.\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "    True\n"
            "    \"\"\""
        )
        clean_prompt = "Construct a verifiable state machine for multi-agent consensus protocols."

        raw_seeds = [
            {"seed_id": "leaked-01", "task_prompt": humaneval_prompt},
            {"seed_id": "clean-02", "task_prompt": clean_prompt},
        ]

        clean, summary, audit = curator.curate_seeds(raw_seeds)

        assert len(clean) == 1
        assert clean[0]["seed_id"] == "clean-02"
        assert summary.contaminated_pruned == 1
        assert "HumanEval" in summary.benchmark_matches
        assert audit[0].status == "CONTAMINATED"

    def test_curate_files_workflow(self, curator, tmp_path: Path):
        in_file = tmp_path / "raw_seeds.jsonl"
        out_file = tmp_path / "clean_seeds.jsonl"
        aud_file = tmp_path / "audit.jsonl"
        sum_file = tmp_path / "summary.json"

        records = [
            {"seed_id": "s1", "role_category": "KERNEL", "task_prompt": "First unique seed prompt for kernel validation."},
            {"seed_id": "s2", "role_category": "KERNEL", "task_prompt": "First unique seed prompt for kernel validation."},
            {"seed_id": "s3", "role_category": "WITNESS", "task_prompt": "Second unique witness prompt for state replay."},
        ]

        with in_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        summary = curator.curate_files([in_file], out_file, audit_file=aud_file, summary_file=sum_file)

        assert summary.total_ingested == 3
        assert summary.final_clean_count == 2
        assert out_file.exists()
        assert aud_file.exists()
        assert sum_file.exists()

        loaded_clean = [json.loads(line) for line in out_file.read_text().splitlines() if line]
        assert len(loaded_clean) == 2
        assert loaded_clean[0]["seed_id"] == "s1"
        assert loaded_clean[1]["seed_id"] == "s3"

    def test_extract_seed_canonical_text(self):
        rec_prompt = {"prompt": "Direct prompt string"}
        assert extract_seed_canonical_text(rec_prompt) == "Direct prompt string"

        rec_scenario = {"scenario": "target_behavior: foo", "role_function": "extract invariants"}
        assert "target_behavior: foo" in extract_seed_canonical_text(rec_scenario)
        assert "extract invariants" in extract_seed_canonical_text(rec_scenario)

        rec_conv = {"messages": [{"role": "user", "content": "hello world"}]}
        assert "content: hello world" in extract_seed_canonical_text(rec_conv)
