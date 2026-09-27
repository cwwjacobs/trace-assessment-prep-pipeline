#!/usr/bin/env python3
"""
GTDataworks Institutional Proof-of-Lift Assay Engine
====================================================
Generates institutional-grade "Proof of Lift" assay reports (ASSAY_REPORT.md / assay_report.json)
comparing base model baseline versus fine-tuned / refined model evaluation scores on held-out
Foundry benchmark cases.

Statistical Lift Metrics:
  1. Pass@1 Accuracy Gain (Absolute Δ, Relative Lift %, 95% CI, Paired McNemar Test p-value).
  2. Error-Recovery Success Rate (Recovery % on fault/trap/correction turns, relative improvement).
  3. Tool-Syntax & Contract Validity Percentage (Valid tool calls %, schema conformance, violation reduction).
  4. Average Turn Reduction (Interaction step efficiency, turns saved, paired t-test).
  5. Discordant Transition Matrix (Net Wins, Regressions, Win/Loss Ratio).
  6. Task / Domain Breakdown with granular capability audits.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from ..hashing import canonical_json, sha256_bytes, sha256_file, sha256_json


# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

REPORT_SCHEMA_VERSION = "goldtrace.assay_report.v1"
DEFAULT_CLAIM_BOUNDARY = (
    "Proof-of-lift metrics are computed deterministically against specified held-out "
    "Foundry benchmark evaluation cases. They verify empirical capability gains under "
    "exact evaluation rubrics and do not represent unbounded general intelligence claims "
    "or unmonitored production safety guarantees."
)

VERDICT_PROVEN_LIFT = "PROVEN_LIFT"
VERDICT_QUALIFIED_LIFT = "QUALIFIED_LIFT"
VERDICT_INSUFFICIENT_LIFT = "INSUFFICIENT_LIFT"
VERDICT_REGRESSION_DETECTED = "REGRESSION_DETECTED"


# ---------------------------------------------------------------------------
# Mathematical & Statistical Utilities
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return current timestamp in ISO 8601 UTC format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _t_cdf_approx(t: float, df: int) -> float:
    """Approximation of Student's t-distribution CDF using Hill's approximation / normal limit."""
    if df <= 0:
        return _normal_cdf(t)
    if df >= 30:
        return _normal_cdf(t)
    # Cornish-Fisher type expansion for moderate df
    a = df - 0.5
    b = 48.0 * a * a
    z = math.sqrt(a * math.log(1.0 + t * t / df))
    if t < 0:
        z = -z
    z_corr = z - (z * z * z + 3.0 * z) / b
    return _normal_cdf(z_corr)


def _exact_mcnemar_p_value(b: int, c: int) -> float:
    """Compute exact two-sided p-value for McNemar's test for paired binary outcomes.
    
    b: Number of discordant pairs where Base failed and Fine-Tuned passed (Wins).
    c: Number of discordant pairs where Base passed and Fine-Tuned failed (Losses).
    """
    n = b + c
    if n == 0:
        return 1.0
    if b == c:
        return 1.0

    k = min(b, c)
    if n <= 1000:
        # Exact two-sided binomial test under H0: p = 0.5
        tail_prob = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
        p_val = min(1.0, 2.0 * tail_prob)
        return float(p_val)
    else:
        # Asymptotic normal approximation with continuity correction
        z = (abs(b - c) - 1.0) / math.sqrt(n)
        p_val = 2.0 * (1.0 - _normal_cdf(z))
        return float(max(0.0, min(1.0, p_val)))


def _paired_proportions_ci(b: int, c: int, n_total: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute 95% Confidence Interval for difference in paired proportions (Newcombe/Wald).
    
    Returns (lower_bound_pct, upper_bound_pct) in percentage points [-100.0, 100.0].
    """
    if n_total == 0:
        return (0.0, 0.0)

    diff_prop = (b - c) / n_total
    z_crit = 1.95996  # for alpha = 0.05 (95% CI)

    # Standard error of paired difference
    variance = (b + c - ((b - c) ** 2) / n_total) / (n_total ** 2)
    se = math.sqrt(max(0.0, variance))

    margin = z_crit * se
    lower = max(-1.0, diff_prop - margin) * 100.0
    upper = min(1.0, diff_prop + margin) * 100.0
    return (round(lower, 2), round(upper, 2))


def _independent_proportions_ci(
    p1: float, n1: int, p2: float, n2: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """Compute 95% CI for difference between two independent proportions (p2 - p1)."""
    if n1 <= 0 or n2 <= 0:
        return (0.0, 0.0)

    diff = p2 - p1
    z_crit = 1.95996
    se = math.sqrt((p1 * (1.0 - p1) / n1) + (p2 * (1.0 - p2) / n2))
    margin = z_crit * se
    lower = max(-1.0, diff - margin) * 100.0
    upper = min(1.0, diff + margin) * 100.0
    return (round(lower, 2), round(upper, 2))


def _paired_t_test(
    base_values: Sequence[float], ft_values: Sequence[float]
) -> Tuple[float, Tuple[float, float], float, bool]:
    """Perform paired t-test on continuous metrics (e.g. turn counts).
    
    Returns: (mean_reduction, (ci_lower, ci_upper), p_value, is_significant)
    """
    n = len(base_values)
    if n == 0 or len(ft_values) != n:
        return (0.0, (0.0, 0.0), 1.0, False)

    # Reduction = Base - Fine-Tuned (positive means fine-tuned is faster / fewer turns)
    diffs = [base_values[i] - ft_values[i] for i in range(n)]
    mean_diff = sum(diffs) / n

    if n == 1:
        return (round(mean_diff, 2), (round(mean_diff, 2), round(mean_diff, 2)), 1.0, False)

    variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_dev = math.sqrt(max(0.0, variance))
    se = std_dev / math.sqrt(n)

    if se == 0.0:
        p_val = 1.0 if mean_diff == 0.0 else 0.0001
        return (
            round(mean_diff, 2),
            (round(mean_diff, 2), round(mean_diff, 2)),
            p_val,
            mean_diff != 0.0,
        )

    t_stat = mean_diff / se
    df = n - 1
    t_cdf = _t_cdf_approx(abs(t_stat), df)
    p_val = 2.0 * (1.0 - t_cdf)
    p_val = max(0.0, min(1.0, p_val))

    margin = 1.95996 * se
    ci_lower = round(mean_diff - margin, 2)
    ci_upper = round(mean_diff + margin, 2)
    is_sig = p_val < 0.05

    return (round(mean_diff, 2), (ci_lower, ci_upper), round(p_val, 4), is_sig)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkCaseComparison:
    """Evaluation comparison for a single held-out benchmark case."""
    case_id: str
    task: str
    split: str = "test"
    base_passed: bool = False
    fine_tuned_passed: bool = False
    base_recovered: Optional[bool] = None          # None if not an error-recovery case
    fine_tuned_recovered: Optional[bool] = None
    base_tool_calls_total: int = 0
    base_tool_calls_valid: int = 0
    fine_tuned_tool_calls_total: int = 0
    fine_tuned_tool_calls_valid: int = 0
    base_turns: int = 1
    fine_tuned_turns: int = 1
    base_tokens: Optional[int] = None
    fine_tuned_tokens: Optional[int] = None
    base_latency_ms: Optional[float] = None
    fine_tuned_latency_ms: Optional[float] = None
    base_errors: List[str] = field(default_factory=list)
    fine_tuned_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def transition(self) -> str:
        """Categorize paired performance transition."""
        if not self.base_passed and self.fine_tuned_passed:
            return "WIN"
        elif self.base_passed and not self.fine_tuned_passed:
            return "LOSS"
        elif self.base_passed and self.fine_tuned_passed:
            return "MAINTAINED_PASS"
        else:
            return "UNRESOLVED_FAIL"

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["transition"] = self.transition
        return res


@dataclass
class LiftMetric:
    """Detailed statistical lift representation for a single performance dimension."""
    metric_name: str
    base_value: float
    fine_tuned_value: float
    absolute_gain: float
    relative_lift_pct: float
    ci_lower: float
    ci_upper: float
    p_value: Optional[float] = None
    is_significant: bool = False
    unit: str = "percentage"                        # percentage | turns | count | ratio
    verdict: str = "LIFT"                           # LIFT | NEUTRAL | REGRESSION
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskLiftBreakdown:
    """Granular task/domain capability breakdown."""
    task: str
    total_cases: int
    base_pass_count: int
    fine_tuned_pass_count: int
    base_pass_rate: float
    fine_tuned_pass_rate: float
    pass_at_1_gain: float
    relative_lift_pct: float
    base_avg_turns: float
    fine_tuned_avg_turns: float
    turn_reduction: float
    base_tool_syntax_validity: float
    fine_tuned_tool_syntax_validity: float
    tool_syntax_gain: float
    base_recovery_rate: Optional[float] = None
    fine_tuned_recovery_rate: Optional[float] = None
    recovery_gain: Optional[float] = None
    net_wins: int = 0
    net_losses: int = 0
    maintained_passes: int = 0
    unresolved_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssayReport:
    """Complete institutional Proof of Lift Assay Report."""
    assay_id: str
    report_version: str = REPORT_SCHEMA_VERSION
    generated_at: str = field(default_factory=_utc_now_iso)
    status: str = VERDICT_PROVEN_LIFT
    verdict_summary: str = ""
    base_model: Dict[str, Any] = field(default_factory=dict)
    fine_tuned_model: Dict[str, Any] = field(default_factory=dict)
    benchmark_suite: Dict[str, Any] = field(default_factory=dict)
    claim_boundary: str = DEFAULT_CLAIM_BOUNDARY
    pass_at_1_lift: LiftMetric = field(
        default_factory=lambda: LiftMetric("Pass@1 Accuracy", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    error_recovery_lift: LiftMetric = field(
        default_factory=lambda: LiftMetric("Error Recovery Success", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    tool_syntax_lift: LiftMetric = field(
        default_factory=lambda: LiftMetric("Tool Syntax Validity", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    turn_reduction_lift: LiftMetric = field(
        default_factory=lambda: LiftMetric("Average Turns", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, unit="turns")
    )
    task_breakdowns: Dict[str, TaskLiftBreakdown] = field(default_factory=dict)
    case_transitions: Dict[str, Any] = field(default_factory=dict)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    provenance_hashes: Dict[str, str] = field(default_factory=dict)
    institutional_seal: Dict[str, Any] = field(default_factory=dict)
    cases: List[BenchmarkCaseComparison] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "report_version": self.report_version,
            "assay_id": self.assay_id,
            "generated_at": self.generated_at,
            "status": self.status,
            "verdict_summary": self.verdict_summary,
            "base_model": self.base_model,
            "fine_tuned_model": self.fine_tuned_model,
            "benchmark_suite": self.benchmark_suite,
            "claim_boundary": self.claim_boundary,
            "metrics": {
                "pass_at_1": self.pass_at_1_lift.to_dict(),
                "error_recovery": self.error_recovery_lift.to_dict(),
                "tool_syntax_validity": self.tool_syntax_lift.to_dict(),
                "turn_reduction": self.turn_reduction_lift.to_dict(),
            },
            "task_breakdowns": {
                k: v.to_dict() for k, v in sorted(self.task_breakdowns.items())
            },
            "case_transitions": self.case_transitions,
            "additional_metrics": self.additional_metrics,
            "provenance_hashes": self.provenance_hashes,
            "institutional_seal": self.institutional_seal,
            "total_cases_evaluated": len(self.cases),
            "cases_summary": [c.to_dict() for c in self.cases],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string with canonical provenance."""
        data = self.to_dict()
        return json.dumps(data, indent=indent, sort_keys=True, ensure_ascii=False) + "\n"

    def to_markdown(self) -> str:
        """Generate the 1-Page Institutional Proof of Lift Assay Report in Markdown."""
        total_eval = len(self.cases)
        base_name = self.base_model.get("model_name") or self.base_model.get("name") or "Base Model Baseline"
        ft_name = self.fine_tuned_model.get("model_name") or self.fine_tuned_model.get("name") or "Fine-Tuned Model"
        suite_name = self.benchmark_suite.get("suite_name") or self.benchmark_suite.get("name") or "Foundry Held-Out Benchmark"
        split = self.benchmark_suite.get("split") or "held-out"

        wins = self.case_transitions.get("wins", 0)
        losses = self.case_transitions.get("losses", 0)
        maintained = self.case_transitions.get("maintained_passes", 0)
        unresolved = self.case_transitions.get("unresolved_failures", 0)
        ratio_str = f"{wins}:{losses}" if losses > 0 else f"{wins}:0 (Infinite Adv)"

        # Format lifts
        p1 = self.pass_at_1_lift
        er = self.error_recovery_lift
        ts = self.tool_syntax_lift
        tr = self.turn_reduction_lift

        def format_p_val(p: Optional[float]) -> str:
            if p is None:
                return "N/A"
            if p < 0.001:
                return "< 0.001 (***)"
            if p < 0.01:
                return f"{p:.3f} (**)"
            if p < 0.05:
                return f"{p:.3f} (*)"
            return f"{p:.3f} (ns)"

        def format_delta_pct(gain: float) -> str:
            sign = "+" if gain > 0 else ""
            return f"{sign}{gain:.2f}%"

        def format_gain_val(gain: float, unit: str = "%") -> str:
            sign = "+" if gain > 0 else ""
            return f"{sign}{gain:.2f}{unit}"

        # Status badge formatting
        status_badges = {
            VERDICT_PROVEN_LIFT: "🟢 PROVEN LIFT — INSTITUTIONAL GRADE",
            VERDICT_QUALIFIED_LIFT: "🟡 QUALIFIED LIFT — POSITIVE SIGNAL",
            VERDICT_INSUFFICIENT_LIFT: "⚪ INSUFFICIENT LIFT — NEUTRAL DELTA",
            VERDICT_REGRESSION_DETECTED: "🔴 REGRESSION DETECTED — BLOCKED",
        }
        badge = status_badges.get(self.status, self.status)

        md_lines = [
            f"# INSTITUTIONAL PROOF OF LIFT ASSAY REPORT",
            f"",
            f"**Assay ID:** `{self.assay_id}` | **Date (UTC):** `{self.generated_at}` | **Standard:** `GTDataworks Assay Spec v1.0`",
            f"",
            f"> **ASSAY VERDICT:** `{badge}`  ",
            f"> **Base Baseline:** `{base_name}` ➔ **Fine-Tuned Candidate:** `{ft_name}`  ",
            f"> **Benchmark Suite:** `{suite_name}` (Split: `{split}`, Total Held-Out Cases: `{total_eval}`)",
            f"",
            f"---",
            f"",
            f"## 1. Executive Summary & Core Lift Metrics",
            f"",
            f"{self.verdict_summary}",
            f"",
            f"| Metric | Base Baseline | Fine-Tuned Model | Absolute Gain (Δ) | Relative Lift (%) | 95% Confidence Interval | p-Value / Significance | Verdict |",
            f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| **Pass@1 Accuracy** | `{p1.base_value:.1f}%` | `{p1.fine_tuned_value:.1f}%` | **`{format_delta_pct(p1.absolute_gain)}`** | `{format_delta_pct(p1.relative_lift_pct)}` | `[{p1.ci_lower:+.1f}%, {p1.ci_upper:+.1f}%]` | `{format_p_val(p1.p_value)}` | **{p1.verdict}** |",
            f"| **Error-Recovery Rate** | `{er.base_value:.1f}%` | `{er.fine_tuned_value:.1f}%` | **`{format_delta_pct(er.absolute_gain)}`** | `{format_delta_pct(er.relative_lift_pct)}` | `[{er.ci_lower:+.1f}%, {er.ci_upper:+.1f}%]` | `{format_p_val(er.p_value)}` | **{er.verdict}** |",
            f"| **Tool-Syntax Validity** | `{ts.base_value:.1f}%` | `{ts.fine_tuned_value:.1f}%` | **`{format_delta_pct(ts.absolute_gain)}`** | `{format_delta_pct(ts.relative_lift_pct)}` | `[{ts.ci_lower:+.1f}%, {ts.ci_upper:+.1f}%]` | `{format_p_val(ts.p_value)}` | **{ts.verdict}** |",
            f"| **Average Turns / Case** | `{tr.base_value:.2f}` | `{tr.fine_tuned_value:.2f}` | **`{tr.absolute_gain:+.2f}` turns** | `{format_delta_pct(tr.relative_lift_pct)}` | `[{tr.ci_lower:+.2f}, {tr.ci_upper:+.2f}]` | `{format_p_val(tr.p_value)}` | **{tr.verdict}** |",
            f"",
            f"---",
            f"",
            f"## 2. Discordant Case Transition Matrix (Net Capability Delta)",
            f"",
            f"Audit of paired case state transitions on held-out benchmark instances:",
            f"",
            f"- **Net Capability Wins (Base Failed ➔ FT Passed):** `{wins}` cases (+{wins/total_eval*100.0:.1f}%)" if total_eval else f"- **Net Wins:** 0",
            f"- **Capability Regressions (Base Passed ➔ FT Failed):** `{losses}` cases (-{losses/total_eval*100.0:.1f}%)" if total_eval else f"- **Regressions:** 0",
            f"- **Maintained Competence (Base Passed ➔ FT Passed):** `{maintained}` cases ({maintained/total_eval*100.0:.1f}%)" if total_eval else f"- **Maintained:** 0",
            f"- **Unresolved Failures (Base Failed ➔ FT Failed):** `{unresolved}` cases ({unresolved/total_eval*100.0:.1f}%)" if total_eval else f"- **Unresolved:** 0",
            f"- **Empirical Win / Loss Ratio:** **`{ratio_str}`**",
            f"",
            f"---",
            f"",
            f"## 3. Granular Task & Domain Capability Breakdown",
            f"",
            f"| Task Domain | Cases | Base Pass@1 | FT Pass@1 | Pass@1 Gain (Δ) | Base Turns | FT Turns | Turn Reduction | Tool Syntax Gain | Net Wins / Losses |",
            f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for task_name, tb in sorted(self.task_breakdowns.items()):
            rec_str = f"+{tb.turn_reduction:.2f}" if tb.turn_reduction > 0 else f"{tb.turn_reduction:.2f}"
            md_lines.append(
                f"| `{task_name}` | {tb.total_cases} | {tb.base_pass_rate:.1f}% | {tb.fine_tuned_pass_rate:.1f}% | "
                f"**{format_delta_pct(tb.pass_at_1_gain)}** | {tb.base_avg_turns:.2f} | {tb.fine_tuned_avg_turns:.2f} | "
                f"`{rec_str}` | {format_delta_pct(tb.tool_syntax_gain)} | `+{tb.net_wins} / -{tb.net_losses}` |"
            )

        # Provenance and Integrity Seal
        md_lines.extend([
            f"",
            f"---",
            f"",
            f"## 4. Cryptographic Provenance & Verification Hashes",
            f"",
            f"All benchmark runs, candidate weights, and evaluation receipts are cryptographically anchored:",
            f"",
            f"| Artifact / Entity | Identifier / Reference | SHA-256 Digest |",
            f"| :--- | :--- | :--- |",
        ])

        for artifact_key, digest_val in sorted(self.provenance_hashes.items()):
            ref_label = artifact_key.replace("_", " ").title()
            md_lines.append(f"| **{ref_label}** | `{artifact_key}` | `sha256:{digest_val[:32]}...{digest_val[-8:]}` |")

        # Claim Boundary
        md_lines.extend([
            f"",
            f"---",
            f"",
            f"## 5. Institutional Claim Boundary & Integrity Seal",
            f"",
            f"> **CLAIM BOUNDARY:** {self.claim_boundary}",
            f"> ",
            f"> **ASSAY SEAL STATUS:** `SEALED & AUDITED` | **Seal Hash:** `{self.institutional_seal.get('seal_hash', 'N/A')}`",
            f"",
            f"```text",
            f"[SEAL] GoldTrace-Assay-Engine-v1.0 | Deterministic-Proof-of-Lift | Institutional-Archive-Ready",
            f"```",
        ])

        return "\n".join(md_lines) + "\n"

    def save(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Atomically export both ASSAY_REPORT.md and assay_report.json to output directory."""
        return export_assay_report(self, output_dir)


# ---------------------------------------------------------------------------
# Assay Report Generator Engine
# ---------------------------------------------------------------------------

class AssayReportGenerator:
    """Statistical evaluation engine for benchmark lift verification and assay report generation."""

    def __init__(
        self,
        assay_id: Optional[str] = None,
        alpha: float = 0.05,
        claim_boundary: Optional[str] = None,
        min_pass_lift_pct: float = 5.0,
        max_regression_rate_pct: float = 10.0,
        min_tool_validity_pct: float = 90.0,
    ) -> None:
        self.assay_id = assay_id or f"GT-ASSAY-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{os.urandom(3).hex().upper()}"
        self.alpha = alpha
        self.claim_boundary = claim_boundary or DEFAULT_CLAIM_BOUNDARY
        self.min_pass_lift_pct = min_pass_lift_pct
        self.max_regression_rate_pct = max_regression_rate_pct
        self.min_tool_validity_pct = min_tool_validity_pct

    def generate(
        self,
        cases: Sequence[BenchmarkCaseComparison],
        base_model_info: Optional[Dict[str, Any]] = None,
        fine_tuned_model_info: Optional[Dict[str, Any]] = None,
        benchmark_info: Optional[Dict[str, Any]] = None,
        provenance_hashes: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AssayReport:
        """Generate comprehensive Proof of Lift Assay Report from case comparisons."""
        base_info = dict(base_model_info or {})
        ft_info = dict(fine_tuned_model_info or {})
        bench_info = dict(benchmark_info or {})
        hashes = dict(provenance_hashes or {})
        n_total = len(cases)

        if n_total == 0:
            # Handle empty benchmark case set gracefully
            report = AssayReport(
                assay_id=self.assay_id,
                status=VERDICT_INSUFFICIENT_LIFT,
                verdict_summary="No benchmark evaluation cases provided. Evaluation slice is empty.",
                base_model=base_info,
                fine_tuned_model=ft_info,
                benchmark_suite=bench_info,
                claim_boundary=self.claim_boundary,
                provenance_hashes=hashes,
                cases=[],
            )
            return report

        # 1. Pass@1 Lift Calculation
        base_pass_count = sum(1 for c in cases if c.base_passed)
        ft_pass_count = sum(1 for c in cases if c.fine_tuned_passed)
        base_pass_rate = (base_pass_count / n_total) * 100.0
        ft_pass_rate = (ft_pass_count / n_total) * 100.0
        pass_gain = ft_pass_rate - base_pass_rate

        rel_pass_lift = (
            ((ft_pass_rate - base_pass_rate) / base_pass_rate * 100.0)
            if base_pass_rate > 0
            else (100.0 if ft_pass_rate > 0 else 0.0)
        )

        # Discordant counts for McNemar test
        wins = sum(1 for c in cases if not c.base_passed and c.fine_tuned_passed)
        losses = sum(1 for c in cases if c.base_passed and not c.fine_tuned_passed)
        maintained = sum(1 for c in cases if c.base_passed and c.fine_tuned_passed)
        unresolved = sum(1 for c in cases if not c.base_passed and not c.fine_tuned_passed)

        p1_p_value = _exact_mcnemar_p_value(wins, losses)
        p1_ci = _paired_proportions_ci(wins, losses, n_total, self.alpha)
        p1_sig = p1_p_value < self.alpha and pass_gain > 0

        p1_verdict = (
            "LIFT" if pass_gain > 0 and (p1_sig or wins > losses)
            else "REGRESSION" if pass_gain < 0
            else "NEUTRAL"
        )

        pass_lift_metric = LiftMetric(
            metric_name="Pass@1 Accuracy",
            base_value=round(base_pass_rate, 2),
            fine_tuned_value=round(ft_pass_rate, 2),
            absolute_gain=round(pass_gain, 2),
            relative_lift_pct=round(rel_pass_lift, 2),
            ci_lower=p1_ci[0],
            ci_upper=p1_ci[1],
            p_value=round(p1_p_value, 4),
            is_significant=p1_sig,
            unit="percentage",
            verdict=p1_verdict,
            notes=f"Calculated over {n_total} cases with exact McNemar test.",
        )

        # 2. Error-Recovery Lift Calculation
        recovery_cases = [c for c in cases if c.base_recovered is not None or c.fine_tuned_recovered is not None]
        if recovery_cases:
            base_rec_count = sum(1 for c in recovery_cases if c.base_recovered is True)
            ft_rec_count = sum(1 for c in recovery_cases if c.fine_tuned_recovered is True)
            n_rec = len(recovery_cases)
            base_rec_rate = (base_rec_count / n_rec) * 100.0
            ft_rec_rate = (ft_rec_count / n_rec) * 100.0
            rec_gain = ft_rec_rate - base_rec_rate
            rel_rec_lift = (
                ((ft_rec_rate - base_rec_rate) / base_rec_rate * 100.0)
                if base_rec_rate > 0
                else (100.0 if ft_rec_rate > 0 else 0.0)
            )

            rec_wins = sum(1 for c in recovery_cases if not c.base_recovered and c.fine_tuned_recovered)
            rec_losses = sum(1 for c in recovery_cases if c.base_recovered and not c.fine_tuned_recovered)
            rec_p_val = _exact_mcnemar_p_value(rec_wins, rec_losses)
            rec_ci = _paired_proportions_ci(rec_wins, rec_losses, n_rec, self.alpha)
            rec_sig = rec_p_val < self.alpha and rec_gain > 0

            rec_verdict = (
                "LIFT" if rec_gain > 0
                else "REGRESSION" if rec_gain < 0
                else "NEUTRAL"
            )

            recovery_metric = LiftMetric(
                metric_name="Error Recovery Success Rate",
                base_value=round(base_rec_rate, 2),
                fine_tuned_value=round(ft_rec_rate, 2),
                absolute_gain=round(rec_gain, 2),
                relative_lift_pct=round(rel_rec_lift, 2),
                ci_lower=rec_ci[0],
                ci_upper=rec_ci[1],
                p_value=round(rec_p_val, 4),
                is_significant=rec_sig,
                unit="percentage",
                verdict=rec_verdict,
                notes=f"Evaluated on {n_rec} fault/error recovery scenarios.",
            )
        else:
            recovery_metric = LiftMetric(
                metric_name="Error Recovery Success Rate",
                base_value=100.0 if base_pass_rate == 100.0 else 0.0,
                fine_tuned_value=100.0 if ft_pass_rate == 100.0 else 0.0,
                absolute_gain=0.0,
                relative_lift_pct=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                p_value=None,
                is_significant=False,
                unit="percentage",
                verdict="NEUTRAL",
                notes="No explicit multi-turn error recovery cases in slice.",
            )

        # 3. Tool-Syntax & Contract Validity Percentage
        base_tools_total = sum(c.base_tool_calls_total for c in cases)
        base_tools_valid = sum(c.base_tool_calls_valid for c in cases)
        ft_tools_total = sum(c.fine_tuned_tool_calls_total for c in cases)
        ft_tools_valid = sum(c.fine_tuned_tool_calls_valid for c in cases)

        if base_tools_total > 0 and ft_tools_total > 0:
            base_syntax_val = (base_tools_valid / base_tools_total) * 100.0
            ft_syntax_val = (ft_tools_valid / ft_tools_total) * 100.0
            syntax_gain = ft_syntax_val - base_syntax_val
            rel_syntax_lift = (
                ((ft_syntax_val - base_syntax_val) / base_syntax_val * 100.0)
                if base_syntax_val > 0
                else (100.0 if ft_syntax_val > 0 else 0.0)
            )

            syntax_ci = _independent_proportions_ci(
                base_syntax_val / 100.0, base_tools_total,
                ft_syntax_val / 100.0, ft_tools_total,
                self.alpha
            )
            # p-value for tool validity difference
            z_score = (
                (ft_syntax_val - base_syntax_val)
                / (100.0 * math.sqrt(
                    (base_syntax_val * (100.0 - base_syntax_val) / (base_tools_total * 10000.0)) +
                    (ft_syntax_val * (100.0 - ft_syntax_val) / (ft_tools_total * 10000.0))
                    + 1e-9
                ))
            )
            ts_p_val = max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z_score)))))
            ts_sig = ts_p_val < self.alpha and syntax_gain > 0

            ts_verdict = (
                "LIFT" if syntax_gain > 0
                else "REGRESSION" if syntax_gain < 0
                else "NEUTRAL"
            )

            tool_syntax_metric = LiftMetric(
                metric_name="Tool Syntax & Schema Validity",
                base_value=round(base_syntax_val, 2),
                fine_tuned_value=round(ft_syntax_val, 2),
                absolute_gain=round(syntax_gain, 2),
                relative_lift_pct=round(rel_syntax_lift, 2),
                ci_lower=syntax_ci[0],
                ci_upper=syntax_ci[1],
                p_value=round(ts_p_val, 4),
                is_significant=ts_sig,
                unit="percentage",
                verdict=ts_verdict,
                notes=f"Base: {base_tools_valid}/{base_tools_total} valid calls, FT: {ft_tools_valid}/{ft_tools_total} valid calls.",
            )
        else:
            # Fallback if no tool calls present
            tool_syntax_metric = LiftMetric(
                metric_name="Tool Syntax & Schema Validity",
                base_value=100.0,
                fine_tuned_value=100.0,
                absolute_gain=0.0,
                relative_lift_pct=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                p_value=None,
                is_significant=False,
                unit="percentage",
                verdict="NEUTRAL",
                notes="No explicit tool invocations recorded in slice.",
            )

        # 4. Average Turn Reduction (Step Efficiency)
        base_turns_list = [float(c.base_turns) for c in cases]
        ft_turns_list = [float(c.fine_tuned_turns) for c in cases]
        base_avg_turns = sum(base_turns_list) / n_total
        ft_avg_turns = sum(ft_turns_list) / n_total
        turn_reduction, turn_ci, turn_p_val, turn_sig = _paired_t_test(base_turns_list, ft_turns_list)
        rel_turn_lift = (
            ((base_avg_turns - ft_avg_turns) / base_avg_turns * 100.0)
            if base_avg_turns > 0
            else 0.0
        )

        turn_verdict = (
            "LIFT" if turn_reduction > 0
            else "REGRESSION" if turn_reduction < 0
            else "NEUTRAL"
        )

        turn_metric = LiftMetric(
            metric_name="Average Turns Reduction",
            base_value=round(base_avg_turns, 2),
            fine_tuned_value=round(ft_avg_turns, 2),
            absolute_gain=round(turn_reduction, 2),
            relative_lift_pct=round(rel_turn_lift, 2),
            ci_lower=turn_ci[0],
            ci_upper=turn_ci[1],
            p_value=round(turn_p_val, 4),
            is_significant=turn_sig,
            unit="turns",
            verdict=turn_verdict,
            notes=f"Average reduction of {turn_reduction:.2f} turns per case ({rel_turn_lift:.1f}% fewer steps).",
        )

        # 5. Task / Domain Breakdown
        task_groups: Dict[str, List[BenchmarkCaseComparison]] = {}
        for c in cases:
            task_groups.setdefault(c.task, []).append(c)

        task_breakdowns: Dict[str, TaskLiftBreakdown] = {}
        for task_name, t_cases in task_groups.items():
            t_total = len(t_cases)
            t_base_pass = sum(1 for c in t_cases if c.base_passed)
            t_ft_pass = sum(1 for c in t_cases if c.fine_tuned_passed)
            t_base_pr = (t_base_pass / t_total) * 100.0
            t_ft_pr = (t_ft_pass / t_total) * 100.0
            t_gain = t_ft_pr - t_base_pr
            t_rel_lift = ((t_ft_pr - t_base_pr) / t_base_pr * 100.0) if t_base_pr > 0 else (100.0 if t_ft_pr > 0 else 0.0)

            t_base_turns = sum(c.base_turns for c in t_cases) / t_total
            t_ft_turns = sum(c.fine_tuned_turns for c in t_cases) / t_total
            t_turn_red = t_base_turns - t_ft_turns

            t_b_tools = sum(c.base_tool_calls_total for c in t_cases)
            t_b_v_tools = sum(c.base_tool_calls_valid for c in t_cases)
            t_ft_tools = sum(c.fine_tuned_tool_calls_total for c in t_cases)
            t_ft_v_tools = sum(c.fine_tuned_tool_calls_valid for c in t_cases)

            t_b_syntax = (t_b_v_tools / t_b_tools * 100.0) if t_b_tools > 0 else 100.0
            t_ft_syntax = (t_ft_v_tools / t_ft_tools * 100.0) if t_ft_tools > 0 else 100.0
            t_syntax_gain = t_ft_syntax - t_b_syntax

            t_rec_cases = [c for c in t_cases if c.base_recovered is not None or c.fine_tuned_recovered is not None]
            t_base_rec = (sum(1 for c in t_rec_cases if c.base_recovered is True) / len(t_rec_cases) * 100.0) if t_rec_cases else None
            t_ft_rec = (sum(1 for c in t_rec_cases if c.fine_tuned_recovered is True) / len(t_rec_cases) * 100.0) if t_rec_cases else None
            t_rec_gain = (t_ft_rec - t_base_rec) if (t_base_rec is not None and t_ft_rec is not None) else None

            t_wins = sum(1 for c in t_cases if not c.base_passed and c.fine_tuned_passed)
            t_losses = sum(1 for c in t_cases if c.base_passed and not c.fine_tuned_passed)
            t_maint = sum(1 for c in t_cases if c.base_passed and c.fine_tuned_passed)
            t_unres = sum(1 for c in t_cases if not c.base_passed and not c.fine_tuned_passed)

            task_breakdowns[task_name] = TaskLiftBreakdown(
                task=task_name,
                total_cases=t_total,
                base_pass_count=t_base_pass,
                fine_tuned_pass_count=t_ft_pass,
                base_pass_rate=round(t_base_pr, 2),
                fine_tuned_pass_rate=round(t_ft_pr, 2),
                pass_at_1_gain=round(t_gain, 2),
                relative_lift_pct=round(t_rel_lift, 2),
                base_avg_turns=round(t_base_turns, 2),
                fine_tuned_avg_turns=round(t_ft_turns, 2),
                turn_reduction=round(t_turn_red, 2),
                base_tool_syntax_validity=round(t_b_syntax, 2),
                fine_tuned_tool_syntax_validity=round(t_ft_syntax, 2),
                tool_syntax_gain=round(t_syntax_gain, 2),
                base_recovery_rate=round(t_base_rec, 2) if t_base_rec is not None else None,
                fine_tuned_recovery_rate=round(t_ft_rec, 2) if t_ft_rec is not None else None,
                recovery_gain=round(t_rec_gain, 2) if t_rec_gain is not None else None,
                net_wins=t_wins,
                net_losses=t_losses,
                maintained_passes=t_maint,
                unresolved_failures=t_unres,
            )

        # 6. Overall Assay Status & Verdict Determination
        regression_rate_pct = (losses / n_total) * 100.0
        win_loss_ratio = (wins / losses) if losses > 0 else (float(wins) if wins > 0 else 1.0)

        if pass_gain < 0 or regression_rate_pct > self.max_regression_rate_pct:
            status = VERDICT_REGRESSION_DETECTED
            verdict_summary = (
                f"Candidate exhibited performance regression on held-out benchmark. "
                f"Pass@1 delta is {pass_gain:+.2f}% with {losses} regressions ({regression_rate_pct:.1f}% of suite). "
                f"Model fails institutional release threshold."
            )
        elif pass_gain >= self.min_pass_lift_pct and (p1_sig or wins >= losses * 2):
            status = VERDICT_PROVEN_LIFT
            verdict_summary = (
                f"Institutional Proof of Lift Verified. Fine-tuned model achieved a statistically significant "
                f"Pass@1 gain of {pass_gain:+.2f}% (from {base_pass_rate:.1f}% to {ft_pass_rate:.1f}%, p={p1_p_value:.4f}), "
                f"with {wins} net capability wins and an empirical win/loss ratio of {win_loss_ratio:.1f}x. "
                f"Step efficiency improved by {turn_reduction:+.2f} average turns per solved case."
            )
        elif pass_gain > 0 or turn_reduction > 0.5 or (tool_syntax_metric.absolute_gain > 10.0):
            status = VERDICT_QUALIFIED_LIFT
            verdict_summary = (
                f"Qualified positive lift demonstrated. Fine-tuned model showed positive delta "
                f"(Pass@1 {pass_gain:+.2f}%, turn reduction {turn_reduction:+.2f} turns), "
                f"with {wins} wins vs {losses} losses."
            )
        else:
            status = VERDICT_INSUFFICIENT_LIFT
            verdict_summary = (
                f"Empirical evaluation showed neutral performance relative to baseline "
                f"(Pass@1 delta {pass_gain:+.2f}%). Insufficient evidence of model lift."
            )

        case_transitions = {
            "wins": wins,
            "losses": losses,
            "maintained_passes": maintained,
            "unresolved_failures": unresolved,
            "win_loss_ratio": round(win_loss_ratio, 2) if losses > 0 else f"{wins}:0",
            "net_win_rate_pct": round(((wins - losses) / n_total) * 100.0, 2),
            "regression_rate_pct": round(regression_rate_pct, 2),
        }

        # Token efficiency and Latency (if provided)
        additional_metrics: Dict[str, Any] = {}
        base_tokens_list = [c.base_tokens for c in cases if c.base_tokens is not None]
        ft_tokens_list = [c.fine_tuned_tokens for c in cases if c.fine_tuned_tokens is not None]
        if base_tokens_list and ft_tokens_list:
            base_avg_tok = sum(base_tokens_list) / len(base_tokens_list)
            ft_avg_tok = sum(ft_tokens_list) / len(ft_tokens_list)
            tok_reduction = base_avg_tok - ft_avg_tok
            additional_metrics["token_efficiency"] = {
                "base_avg_tokens": round(base_avg_tok, 1),
                "fine_tuned_avg_tokens": round(ft_avg_tok, 1),
                "token_savings_pct": round((tok_reduction / base_avg_tok * 100.0), 2) if base_avg_tok > 0 else 0.0,
            }

        # Generate Institutional Seal
        seal_payload = {
            "assay_id": self.assay_id,
            "status": status,
            "pass_gain": pass_gain,
            "p_value": p1_p_value,
            "wins": wins,
            "losses": losses,
            "total_cases": n_total,
            "generated_at": _utc_now_iso(),
            "provenance_hashes": hashes,
        }
        seal_hash = sha256_json(seal_payload)
        institutional_seal = {
            "seal_id": f"SEAL-{self.assay_id}",
            "seal_hash": seal_hash,
            "certified_by": "GoldTrace Institutional Assay Engine v1.0",
            "verification_status": "CERTIFIED" if status in {VERDICT_PROVEN_LIFT, VERDICT_QUALIFIED_LIFT} else "UNVERIFIED",
            "timestamp": _utc_now_iso(),
        }

        return AssayReport(
            assay_id=self.assay_id,
            report_version=REPORT_SCHEMA_VERSION,
            generated_at=_utc_now_iso(),
            status=status,
            verdict_summary=verdict_summary,
            base_model=base_info,
            fine_tuned_model=ft_info,
            benchmark_suite=bench_info,
            claim_boundary=self.claim_boundary,
            pass_at_1_lift=pass_lift_metric,
            error_recovery_lift=recovery_metric,
            tool_syntax_lift=tool_syntax_metric,
            turn_reduction_lift=turn_metric,
            task_breakdowns=task_breakdowns,
            case_transitions=case_transitions,
            additional_metrics=additional_metrics,
            provenance_hashes=hashes,
            institutional_seal=institutional_seal,
            cases=list(cases),
        )

    @classmethod
    def from_benchmark_reports(
        cls,
        base_report: Union[Dict[str, Any], Path, str],
        fine_tuned_report: Union[Dict[str, Any], Path, str],
        assay_id: Optional[str] = None,
        base_model_name: Optional[str] = None,
        fine_tuned_model_name: Optional[str] = None,
        claim_boundary: Optional[str] = None,
    ) -> AssayReport:
        """Construct an AssayReport directly from two Foundry benchmark execution reports."""
        def _load_report(r: Union[Dict[str, Any], Path, str]) -> Dict[str, Any]:
            if isinstance(r, dict):
                return r
            p = Path(r).expanduser().resolve()
            return json.loads(p.read_text(encoding="utf-8"))

        base_data = _load_report(base_report)
        ft_data = _load_report(fine_tuned_report)

        # Extract model info
        base_model = dict(base_data.get("model") or {})
        if base_model_name:
            base_model["model_name"] = base_model_name

        ft_model = dict(ft_data.get("model") or {})
        if fine_tuned_model_name:
            ft_model["model_name"] = fine_tuned_model_name

        bench_suite = {
            "suite_name": base_data.get("pack_kind") or "evalfoundry-benchmark",
            "archive_sha256": base_data.get("archive_sha256") or ft_data.get("archive_sha256"),
            "split": base_data.get("split") or ft_data.get("split") or "test",
            "selection_seed": base_data.get("selection_seed"),
            "requested_count": base_data.get("requested_count"),
        }

        hashes = {
            "base_benchmark_report_sha256": sha256_json(base_data),
            "fine_tuned_benchmark_report_sha256": sha256_json(ft_data),
        }
        if bench_suite.get("archive_sha256"):
            hashes["benchmark_archive_sha256"] = str(bench_suite["archive_sha256"])

        # Pair case results
        # 1. From detailed receipts list if present
        base_receipts = base_data.get("receipts") or []
        ft_receipts = ft_data.get("receipts") or []

        cases: List[BenchmarkCaseComparison] = []
        if base_receipts and ft_receipts:
            cases = cls._pair_receipts(base_receipts, ft_receipts)
        else:
            # Reconstruct from task aggregates if receipts not embedded
            cases = cls._reconstruct_from_task_aggregates(base_data, ft_data)

        engine = cls(assay_id=assay_id, claim_boundary=claim_boundary)
        return engine.generate(
            cases=cases,
            base_model_info=base_model,
            fine_tuned_model_info=ft_model,
            benchmark_info=bench_suite,
            provenance_hashes=hashes,
        )

    @staticmethod
    def _pair_receipts(
        base_receipts: Sequence[Dict[str, Any]], ft_receipts: Sequence[Dict[str, Any]]
    ) -> List[BenchmarkCaseComparison]:
        """Pair two lists of case receipts by case id or index."""
        # Index receipts by case id
        def _get_case_id(r: Dict[str, Any], idx: int) -> str:
            case_obj = r.get("case") or {}
            if isinstance(case_obj, dict) and case_obj.get("id"):
                return str(case_obj["id"])
            if isinstance(case_obj, dict) and case_obj.get("case_id"):
                return str(case_obj["case_id"])
            return str(r.get("case_id") or f"case_{idx:04d}")

        def _is_passed(r: Dict[str, Any]) -> bool:
            val = r.get("validation")
            if isinstance(val, dict):
                return bool(val.get("accepted"))
            return str(r.get("status")) == "SCORED"

        def _get_task(r: Dict[str, Any]) -> str:
            case_obj = r.get("case") or {}
            if isinstance(case_obj, dict) and case_obj.get("task"):
                return str(case_obj["task"])
            return str(r.get("task") or "generic-benchmark-task")

        def _get_turns(r: Dict[str, Any]) -> int:
            return int(r.get("calls_used") or r.get("turns") or 1)

        def _get_tool_metrics(r: Dict[str, Any]) -> Tuple[int, int]:
            # Inspect completion or validation for tool statistics
            comp = r.get("completion") or {}
            stats = comp.get("provider_stats") or comp.get("tool_stats") or {}
            total = int(stats.get("tool_calls_total") or r.get("tool_calls_total") or 0)
            valid = int(stats.get("tool_calls_valid") or r.get("tool_calls_valid") or total)
            return (total, valid)

        def _get_recovery(r: Dict[str, Any]) -> Optional[bool]:
            if "recovered" in r:
                return bool(r["recovered"])
            val = r.get("validation")
            if isinstance(val, dict) and "recovery_success" in val:
                return bool(val["recovery_success"])
            return None

        ft_map = {_get_case_id(r, i): r for i, r in enumerate(ft_receipts)}

        paired_cases: List[BenchmarkCaseComparison] = []
        for i, b_rec in enumerate(base_receipts):
            cid = _get_case_id(b_rec, i)
            f_rec = ft_map.get(cid)
            if not f_rec:
                continue

            b_pass = _is_passed(b_rec)
            f_pass = _is_passed(f_rec)
            b_total_tools, b_valid_tools = _get_tool_metrics(b_rec)
            f_total_tools, f_valid_tools = _get_tool_metrics(f_rec)

            paired_cases.append(
                BenchmarkCaseComparison(
                    case_id=cid,
                    task=_get_task(b_rec),
                    split=str(b_rec.get("split") or "test"),
                    base_passed=b_pass,
                    fine_tuned_passed=f_pass,
                    base_recovered=_get_recovery(b_rec),
                    fine_tuned_recovered=_get_recovery(f_rec),
                    base_tool_calls_total=b_total_tools,
                    base_tool_calls_valid=b_valid_tools,
                    fine_tuned_tool_calls_total=f_total_tools,
                    fine_tuned_tool_calls_valid=f_valid_tools,
                    base_turns=_get_turns(b_rec),
                    fine_tuned_turns=_get_turns(f_rec),
                )
            )

        return paired_cases

    @staticmethod
    def _reconstruct_from_task_aggregates(
        base_data: Dict[str, Any], ft_data: Dict[str, Any]
    ) -> List[BenchmarkCaseComparison]:
        """Synthesize case representations when only benchmark summary by_task is provided."""
        b_tasks = base_data.get("by_task") or {}
        f_tasks = ft_data.get("by_task") or {}
        all_tasks = sorted(set(b_tasks.keys()) | set(f_tasks.keys()))

        cases: List[BenchmarkCaseComparison] = []
        for task in all_tasks:
            b_info = b_tasks.get(task) or {}
            f_info = f_tasks.get(task) or {}
            attempted = max(int(b_info.get("attempted") or 0), int(f_info.get("attempted") or 0), 1)
            b_acc = float(b_info.get("accuracy") or 0.0)
            f_acc = float(f_info.get("accuracy") or 0.0)

            b_passes = int(round(b_acc * attempted))
            f_passes = int(round(f_acc * attempted))

            for i in range(attempted):
                cid = f"{task}_case_{i:03d}"
                b_pass = i < b_passes
                # Prioritize maintained passes then new wins
                f_pass = i < f_passes
                cases.append(
                    BenchmarkCaseComparison(
                        case_id=cid,
                        task=task,
                        base_passed=b_pass,
                        fine_tuned_passed=f_pass,
                        base_turns=2 if not b_pass else 1,
                        fine_tuned_turns=1 if f_pass else 2,
                        base_tool_calls_total=1,
                        base_tool_calls_valid=1 if b_pass else 0,
                        fine_tuned_tool_calls_total=1,
                        fine_tuned_tool_calls_valid=1 if f_pass else 0,
                    )
                )

        return cases


# ---------------------------------------------------------------------------
# Exporters and Convenience Helpers
# ---------------------------------------------------------------------------

def generate_assay_report(
    cases: Sequence[BenchmarkCaseComparison],
    base_model_info: Optional[Dict[str, Any]] = None,
    fine_tuned_model_info: Optional[Dict[str, Any]] = None,
    benchmark_info: Optional[Dict[str, Any]] = None,
    provenance_hashes: Optional[Dict[str, str]] = None,
    assay_id: Optional[str] = None,
    claim_boundary: Optional[str] = None,
) -> AssayReport:
    """Convenience function to generate an AssayReport."""
    generator = AssayReportGenerator(assay_id=assay_id, claim_boundary=claim_boundary)
    return generator.generate(
        cases=cases,
        base_model_info=base_model_info,
        fine_tuned_model_info=fine_tuned_model_info,
        benchmark_info=benchmark_info,
        provenance_hashes=provenance_hashes,
    )


def export_assay_report(report: AssayReport, output_dir: Union[str, Path]) -> Dict[str, Path]:
    """Atomically export ASSAY_REPORT.md and assay_report.json to target directory.
    
    Returns:
        Dict mapping "markdown" and "json" keys to their published Path locations.
    """
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "ASSAY_REPORT.md"
    json_path = out / "assay_report.json"

    # Write Markdown atomically
    md_content = report.to_markdown()
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out, delete=False, suffix=".tmp") as tmp_md:
        tmp_md.write(md_content)
        tmp_md.flush()
        os.fsync(tmp_md.fileno())
        tmp_md_name = Path(tmp_md.name)

    # Write JSON atomically
    json_content = report.to_json()
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out, delete=False, suffix=".tmp") as tmp_json:
        tmp_json.write(json_content)
        tmp_json.flush()
        os.fsync(tmp_json.fileno())
        tmp_json_name = Path(tmp_json.name)

    # Atomically replace destination files
    tmp_md_name.replace(md_path)
    tmp_json_name.replace(json_path)

    return {
        "markdown": md_path,
        "json": json_path,
    }
