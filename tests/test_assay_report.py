#!/usr/bin/env python3
"""
Unit and Integration Test Suite for GTDataworks Institutional Proof-of-Lift Assay Engine.
========================================================================================
Validates:
  - Exact McNemar paired test and confidence intervals.
  - Pass@1 accuracy gain and statistical significance.
  - Error-recovery success rate lift across fault scenarios.
  - Tool-syntax & contract validity percentage gains.
  - Average turn reduction (step-efficiency paired t-test).
  - Task/domain capability breakdowns and discordant transition matrix.
  - Direct ingestion from Foundry benchmark reports and raw receipts.
  - 1-page institutional ASSAY_REPORT.md and assay_report.json generation.
  - Atomic export and cryptographic seal verification.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

from goldtrace_mint.factory.assay_report import (
    DEFAULT_CLAIM_BOUNDARY,
    REPORT_SCHEMA_VERSION,
    VERDICT_INSUFFICIENT_LIFT,
    VERDICT_PROVEN_LIFT,
    VERDICT_QUALIFIED_LIFT,
    VERDICT_REGRESSION_DETECTED,
    AssayReport,
    AssayReportGenerator,
    BenchmarkCaseComparison,
    LiftMetric,
    TaskLiftBreakdown,
    _exact_mcnemar_p_value,
    _independent_proportions_ci,
    _paired_proportions_ci,
    _paired_t_test,
    _t_cdf_approx,
    export_assay_report,
    generate_assay_report,
)


# ---------------------------------------------------------------------------
# Test Fixtures & Helper Generators
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_lift_cases() -> List[BenchmarkCaseComparison]:
    """Synthetic held-out benchmark evaluation suite with clear positive lift."""
    cases = []
    # 20 cases of tool-contract-v1: Base 50% pass, FT 95% pass (9 wins, 0 losses)
    for i in range(20):
        base_pass = i < 10
        ft_pass = i < 19
        cases.append(
            BenchmarkCaseComparison(
                case_id=f"tool_case_{i:03d}",
                task="tool-contract-v1",
                split="held-out",
                base_passed=base_pass,
                fine_tuned_passed=ft_pass,
                base_tool_calls_total=2,
                base_tool_calls_valid=1 if base_pass else 0,
                fine_tuned_tool_calls_total=2,
                fine_tuned_tool_calls_valid=2 if ft_pass else 1,
                base_turns=3 if not base_pass else 2,
                fine_tuned_turns=1 if ft_pass else 2,
                base_tokens=450,
                fine_tuned_tokens=220,
            )
        )

    # 15 cases of error-recovery-v1: Base 20% recovery, FT 86.7% recovery
    for i in range(15):
        base_rec = i < 3
        ft_rec = i < 13
        cases.append(
            BenchmarkCaseComparison(
                case_id=f"recovery_case_{i:03d}",
                task="error-recovery-v1",
                split="held-out",
                base_passed=base_rec,
                fine_tuned_passed=ft_rec,
                base_recovered=base_rec,
                fine_tuned_recovered=ft_rec,
                base_tool_calls_total=3,
                base_tool_calls_valid=2,
                fine_tuned_tool_calls_total=3,
                fine_tuned_tool_calls_valid=3,
                base_turns=4,
                fine_tuned_turns=2,
                base_tokens=600,
                fine_tuned_tokens=310,
            )
        )

    # 15 cases of policy-shall-gate-v1: Base 60% pass, FT 93.3% pass
    for i in range(15):
        base_pass = i < 9
        ft_pass = i < 14
        cases.append(
            BenchmarkCaseComparison(
                case_id=f"policy_case_{i:03d}",
                task="policy-shall-gate-v1",
                split="held-out",
                base_passed=base_pass,
                fine_tuned_passed=ft_pass,
                base_tool_calls_total=1,
                base_tool_calls_valid=1 if base_pass else 0,
                fine_tuned_tool_calls_total=1,
                fine_tuned_tool_calls_valid=1 if ft_pass else 0,
                base_turns=2,
                fine_tuned_turns=1,
                base_tokens=300,
                fine_tuned_tokens=180,
            )
        )

    return cases


# ---------------------------------------------------------------------------
# 1. Statistical Math & Significance Tests
# ---------------------------------------------------------------------------

class TestStatisticalUtilities:
    def test_exact_mcnemar_p_value_symmetric(self):
        # Equal discordant pairs -> p = 1.0
        assert _exact_mcnemar_p_value(5, 5) == 1.0
        assert _exact_mcnemar_p_value(0, 0) == 1.0

    def test_exact_mcnemar_p_value_extreme_lift(self):
        # 20 wins, 0 losses -> extremely small p-value (< 0.0001)
        p_val = _exact_mcnemar_p_value(20, 0)
        assert p_val < 0.0001
        assert p_val > 0.0

    def test_exact_mcnemar_p_value_moderate_lift(self):
        # 12 wins, 2 losses -> significant lift
        p_val = _exact_mcnemar_p_value(12, 2)
        assert p_val < 0.05
        assert p_val > 0.001

    def test_exact_mcnemar_large_sample_asymptotic(self):
        # Large n (> 1000) calls asymptotic branch
        p_val = _exact_mcnemar_p_value(800, 300)
        assert p_val < 0.00001
        assert 0.0 <= p_val <= 1.0

    def test_paired_proportions_ci(self):
        # 20 wins, 0 losses in N=100 -> diff = 20%, CI should be around [12.16%, 27.84%]
        lower, upper = _paired_proportions_ci(20, 0, 100)
        assert lower > 10.0
        assert upper < 30.0
        assert lower < upper

    def test_paired_proportions_ci_empty(self):
        assert _paired_proportions_ci(0, 0, 0) == (0.0, 0.0)

    def test_independent_proportions_ci(self):
        # p1 = 0.5 (50/100), p2 = 0.9 (90/100) -> diff = 40%
        lower, upper = _independent_proportions_ci(0.5, 100, 0.9, 100)
        assert lower > 25.0
        assert upper < 55.0

    def test_paired_t_test_turn_reduction(self):
        base_turns = [4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 4.0, 3.0]
        ft_turns = [2.0, 2.0, 1.0, 2.0, 2.0, 3.0, 2.0, 1.0]
        mean_red, (ci_low, ci_high), p_val, is_sig = _paired_t_test(base_turns, ft_turns)

        assert mean_red > 2.0
        assert ci_low > 1.5
        assert ci_high < 3.5
        assert p_val < 0.001
        assert is_sig is True

    def test_paired_t_test_identical_turns(self):
        turns = [2.0, 2.0, 2.0, 2.0]
        mean_red, (ci_low, ci_high), p_val, is_sig = _paired_t_test(turns, turns)
        assert mean_red == 0.0
        assert p_val == 1.0
        assert is_sig is False

    def test_paired_t_test_single_value(self):
        mean_red, (ci_low, ci_high), p_val, is_sig = _paired_t_test([3.0], [1.0])
        assert mean_red == 2.0
        assert p_val == 1.0
        assert is_sig is False


# ---------------------------------------------------------------------------
# 2. BenchmarkCaseComparison Data Model Tests
# ---------------------------------------------------------------------------

class TestBenchmarkCaseComparison:
    def test_transition_win(self):
        case = BenchmarkCaseComparison(
            case_id="case_01",
            task="task_a",
            base_passed=False,
            fine_tuned_passed=True,
        )
        assert case.transition == "WIN"
        data = case.to_dict()
        assert data["transition"] == "WIN"
        assert data["base_passed"] is False
        assert data["fine_tuned_passed"] is True

    def test_transition_loss(self):
        case = BenchmarkCaseComparison(
            case_id="case_02",
            task="task_a",
            base_passed=True,
            fine_tuned_passed=False,
        )
        assert case.transition == "LOSS"

    def test_transition_maintained_and_unresolved(self):
        c_maint = BenchmarkCaseComparison("c1", "t1", base_passed=True, fine_tuned_passed=True)
        assert c_maint.transition == "MAINTAINED_PASS"

        c_unres = BenchmarkCaseComparison("c2", "t1", base_passed=False, fine_tuned_passed=False)
        assert c_unres.transition == "UNRESOLVED_FAIL"


# ---------------------------------------------------------------------------
# 3. AssayReportGenerator Core Lift Verification Tests
# ---------------------------------------------------------------------------

class TestAssayReportGenerator:
    def test_generate_proven_lift_report(self, sample_lift_cases):
        generator = AssayReportGenerator(
            assay_id="TEST-ASSAY-001",
            min_pass_lift_pct=5.0,
            max_regression_rate_pct=10.0,
        )

        base_model_info = {
            "model_name": "Qwen2.5-Coder-7B-Base",
            "checkpoint_sha256": "11112222333344445555666677778888",
        }
        ft_model_info = {
            "model_name": "GoldTrace-Refined-Qwen2.5-Coder-7B",
            "lot_id": "LOT-2026-08-001",
            "recipe_id": "sft-loss-masked-v1",
        }
        bench_info = {
            "suite_name": "Foundry-AgentOps-HeldOut-v1",
            "split": "held-out",
            "archive_sha256": "abcdef1234567890abcdef1234567890",
        }
        provenance = {
            "base_run_sha256": "aaaa1111",
            "fine_tuned_run_sha256": "bbbb2222",
            "training_lot_sha256": "cccc3333",
        }

        report = generator.generate(
            cases=sample_lift_cases,
            base_model_info=base_model_info,
            fine_tuned_model_info=ft_model_info,
            benchmark_info=bench_info,
            provenance_hashes=provenance,
        )

        # Verify Top-Level Structure
        assert report.assay_id == "TEST-ASSAY-001"
        assert report.status == VERDICT_PROVEN_LIFT
        assert report.report_version == REPORT_SCHEMA_VERSION
        assert len(report.cases) == 50

        # Verify Pass@1 Metric
        p1 = report.pass_at_1_lift
        assert p1.metric_name == "Pass@1 Accuracy"
        # Total cases = 50. Base pass = 10 (tool) + 3 (recovery) + 9 (policy) = 22/50 = 44.0%
        # FT pass = 19 (tool) + 13 (recovery) + 14 (policy) = 46/50 = 92.0%
        assert p1.base_value == 44.0
        assert p1.fine_tuned_value == 92.0
        assert p1.absolute_gain == 48.0
        assert p1.relative_lift_pct == pytest.approx((48.0 / 44.0) * 100.0, rel=1e-2)
        assert p1.p_value is not None and p1.p_value < 0.001
        assert p1.is_significant is True
        assert p1.verdict == "LIFT"

        # Verify Error Recovery Metric
        er = report.error_recovery_lift
        assert er.metric_name == "Error Recovery Success Rate"
        # 15 recovery cases. Base = 3/15 (20.0%), FT = 13/15 (86.67%)
        assert er.base_value == 20.0
        assert er.fine_tuned_value == pytest.approx(86.67, abs=0.1)
        assert er.absolute_gain > 60.0
        assert er.verdict == "LIFT"

        # Verify Tool Syntax Validity Metric
        ts = report.tool_syntax_lift
        assert ts.metric_name == "Tool Syntax & Schema Validity"
        assert ts.base_value < ts.fine_tuned_value
        assert ts.absolute_gain > 0.0
        assert ts.verdict == "LIFT"

        # Verify Turn Reduction Metric
        tr = report.turn_reduction_lift
        assert tr.unit == "turns"
        assert tr.base_value > tr.fine_tuned_value
        assert tr.absolute_gain > 1.0  # At least 1 full turn saved on average
        assert tr.p_value is not None and tr.p_value < 0.001
        assert tr.verdict == "LIFT"

        # Verify Case Transitions & Win/Loss Matrix
        transitions = report.case_transitions
        assert transitions["wins"] == 24  # 9 (tool) + 10 (recovery) + 5 (policy) = 24 wins
        assert transitions["losses"] == 0
        assert transitions["maintained_passes"] == 22
        assert transitions["unresolved_failures"] == 4
        assert transitions["regression_rate_pct"] == 0.0

        # Verify Task Breakdowns
        assert "tool-contract-v1" in report.task_breakdowns
        assert "error-recovery-v1" in report.task_breakdowns
        assert "policy-shall-gate-v1" in report.task_breakdowns

        tool_tb = report.task_breakdowns["tool-contract-v1"]
        assert tool_tb.total_cases == 20
        assert tool_tb.base_pass_rate == 50.0
        assert tool_tb.fine_tuned_pass_rate == 95.0
        assert tool_tb.pass_at_1_gain == 45.0
        assert tool_tb.net_wins == 9
        assert tool_tb.net_losses == 0

        # Verify Additional Metrics
        assert "token_efficiency" in report.additional_metrics
        tok_eff = report.additional_metrics["token_efficiency"]
        assert tok_eff["token_savings_pct"] > 30.0

        # Verify Institutional Seal
        assert report.institutional_seal["verification_status"] == "CERTIFIED"
        assert len(report.institutional_seal["seal_hash"]) == 64

    def test_generate_regression_detected_report(self):
        # Test candidate that regressed significantly
        cases = []
        for i in range(20):
            # Base passed 18, FT passed only 5 -> 13 regressions
            b_pass = i < 18
            f_pass = i < 5
            cases.append(
                BenchmarkCaseComparison(
                    case_id=f"reg_case_{i:03d}",
                    task="tool-contract-v1",
                    base_passed=b_pass,
                    fine_tuned_passed=f_pass,
                    base_turns=1,
                    fine_tuned_turns=3,
                )
            )

        generator = AssayReportGenerator(assay_id="REGRESSION-ASSAY-001")
        report = generator.generate(cases)

        assert report.status == VERDICT_REGRESSION_DETECTED
        assert report.pass_at_1_lift.absolute_gain < 0
        assert report.pass_at_1_lift.verdict == "REGRESSION"
        assert report.case_transitions["losses"] == 13
        assert report.institutional_seal["verification_status"] == "UNVERIFIED"

    def test_generate_empty_cases_graceful_handling(self):
        generator = AssayReportGenerator(assay_id="EMPTY-ASSAY")
        report = generator.generate([])

        assert report.status == VERDICT_INSUFFICIENT_LIFT
        assert len(report.cases) == 0
        assert "empty" in report.verdict_summary.lower()


# ---------------------------------------------------------------------------
# 4. Ingestion From Foundry Benchmark Reports Tests
# ---------------------------------------------------------------------------

class TestFoundryBenchmarkReportIngestion:
    def test_from_benchmark_reports_with_receipts(self, tmp_path):
        base_receipts = [
            {
                "run_id": f"base_run_{i}",
                "case": {"id": f"c_{i}", "task": "tool-contract-v1"},
                "status": "SCORED",
                "validation": {"accepted": i < 5},
                "calls_used": 3,
                "completion": {"tool_stats": {"tool_calls_total": 2, "tool_calls_valid": 1 if i < 5 else 0}},
            }
            for i in range(10)
        ]

        ft_receipts = [
            {
                "run_id": f"ft_run_{i}",
                "case": {"id": f"c_{i}", "task": "tool-contract-v1"},
                "status": "SCORED",
                "validation": {"accepted": i < 9},
                "calls_used": 1,
                "completion": {"tool_stats": {"tool_calls_total": 2, "tool_calls_valid": 2}},
            }
            for i in range(10)
        ]

        base_report_data = {
            "report_version": "evalfoundry-benchmark-v1",
            "pack_kind": "tool_contract_public_v1",
            "archive_sha256": "feedbeef123456",
            "split": "test",
            "model": {"endpoint": "http://localhost:8000", "model": "base-model"},
            "receipts": base_receipts,
        }

        ft_report_data = {
            "report_version": "evalfoundry-benchmark-v1",
            "pack_kind": "tool_contract_public_v1",
            "archive_sha256": "feedbeef123456",
            "split": "test",
            "model": {"endpoint": "http://localhost:8000", "model": "refined-model"},
            "receipts": ft_receipts,
        }

        base_file = tmp_path / "base_benchmark.json"
        ft_file = tmp_path / "ft_benchmark.json"
        base_file.write_text(json.dumps(base_report_data), encoding="utf-8")
        ft_file.write_text(json.dumps(ft_report_data), encoding="utf-8")

        report = AssayReportGenerator.from_benchmark_reports(
            base_report=base_file,
            fine_tuned_report=ft_file,
            base_model_name="Base-Model-7B",
            fine_tuned_model_name="FineTuned-Model-7B",
        )

        assert report.status in {VERDICT_PROVEN_LIFT, VERDICT_QUALIFIED_LIFT}
        assert report.pass_at_1_lift.base_value == 50.0
        assert report.pass_at_1_lift.fine_tuned_value == 90.0
        assert report.pass_at_1_lift.absolute_gain == 40.0
        assert report.turn_reduction_lift.absolute_gain == 2.0  # From 3 to 1 turns
        assert report.base_model["model_name"] == "Base-Model-7B"
        assert report.fine_tuned_model["model_name"] == "FineTuned-Model-7B"

    def test_from_benchmark_reports_task_aggregates_fallback(self):
        base_data = {
            "report_version": "evalfoundry-benchmark-v1",
            "pack_kind": "agent_ops_public_v1",
            "by_task": {
                "action-gate-v1": {"attempted": 20, "accepted": 10, "accuracy": 0.50},
                "skill-pack-checklist-v1": {"attempted": 20, "accepted": 12, "accuracy": 0.60},
            },
        }
        ft_data = {
            "report_version": "evalfoundry-benchmark-v1",
            "pack_kind": "agent_ops_public_v1",
            "by_task": {
                "action-gate-v1": {"attempted": 20, "accepted": 18, "accuracy": 0.90},
                "skill-pack-checklist-v1": {"attempted": 20, "accepted": 19, "accuracy": 0.95},
            },
        }

        report = AssayReportGenerator.from_benchmark_reports(
            base_report=base_data,
            fine_tuned_report=ft_data,
        )

        assert report.status == VERDICT_PROVEN_LIFT
        assert len(report.cases) == 40
        assert report.pass_at_1_lift.absolute_gain == pytest.approx(37.5, abs=0.5)


# ---------------------------------------------------------------------------
# 5. Output Formatting & Export Tests
# ---------------------------------------------------------------------------

class TestAssayReportOutputs:
    def test_to_markdown_formatting(self, sample_lift_cases):
        report = generate_assay_report(
            cases=sample_lift_cases,
            base_model_info={"model_name": "Base-Qwen"},
            fine_tuned_model_info={"model_name": "FT-Qwen"},
            assay_id="GT-ASSAY-MD-001",
        )

        md = report.to_markdown()

        # Check required institutional sections
        assert "# INSTITUTIONAL PROOF OF LIFT ASSAY REPORT" in md
        assert "**Assay ID:** `GT-ASSAY-MD-001`" in md
        assert "## 1. Executive Summary & Core Lift Metrics" in md
        assert "## 2. Discordant Case Transition Matrix" in md
        assert "## 3. Granular Task & Domain Capability Breakdown" in md
        assert "## 4. Cryptographic Provenance & Verification Hashes" in md
        assert "## 5. Institutional Claim Boundary & Integrity Seal" in md

        # Check table headers and contents
        assert "| **Pass@1 Accuracy** |" in md
        assert "| **Error-Recovery Rate** |" in md
        assert "| **Tool-Syntax Validity** |" in md
        assert "| **Average Turns / Case** |" in md
        assert "`tool-contract-v1`" in md
        assert "`error-recovery-v1`" in md
        assert "`policy-shall-gate-v1`" in md
        assert "CLAIM BOUNDARY:" in md
        assert "ASSAY SEAL STATUS:" in md

    def test_to_json_validity(self, sample_lift_cases):
        report = generate_assay_report(
            cases=sample_lift_cases,
            assay_id="GT-ASSAY-JSON-001",
        )

        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert parsed["assay_id"] == "GT-ASSAY-JSON-001"
        assert parsed["report_version"] == REPORT_SCHEMA_VERSION
        assert "metrics" in parsed
        assert "pass_at_1" in parsed["metrics"]
        assert "error_recovery" in parsed["metrics"]
        assert "tool_syntax_validity" in parsed["metrics"]
        assert "turn_reduction" in parsed["metrics"]
        assert parsed["total_cases_evaluated"] == 50
        assert len(parsed["cases_summary"]) == 50

    def test_atomic_export_to_directory(self, tmp_path, sample_lift_cases):
        report = generate_assay_report(
            cases=sample_lift_cases,
            assay_id="GT-ASSAY-EXPORT-001",
        )

        output_dir = tmp_path / "assay_output"
        published_files = export_assay_report(report, output_dir)

        assert "markdown" in published_files
        assert "json" in published_files

        md_file = published_files["markdown"]
        json_file = published_files["json"]

        assert md_file.exists() and md_file.name == "ASSAY_REPORT.md"
        assert json_file.exists() and json_file.name == "assay_report.json"

        # Verify contents of exported files
        json_content = json.loads(json_file.read_text(encoding="utf-8"))
        assert json_content["assay_id"] == "GT-ASSAY-EXPORT-001"

        md_content = md_file.read_text(encoding="utf-8")
        assert "INSTITUTIONAL PROOF OF LIFT ASSAY REPORT" in md_content

        # Test report.save() shorthand
        save_dir = tmp_path / "save_output"
        res = report.save(save_dir)
        assert res["markdown"].exists()
        assert res["json"].exists()
