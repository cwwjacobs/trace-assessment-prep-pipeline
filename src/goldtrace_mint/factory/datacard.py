"""Institutional DataCard & Metadata Architect Module for Gold-Trace-Dataworks.

Implements automated generation of institutional DATACARD.md and datacard.json
per the Gebru et al. (Datasheets for Datasets) standard, computes token statistics,
and records cryptographic lot hashes, task taxonomy, license certification,
and provenance chains.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from goldtrace_mint.hashing import sha256_file, sha256_json, sha256_product_lot


@dataclass
class TokenStatistics:
    """Statistical summary of token distribution and interaction turns across dataset records."""

    total_records: int = 0
    total_tokens: int = 0
    mean_prompt_length: float = 0.0
    mean_cot_reasoning_tokens: float = 0.0
    mean_tool_execution_turns: float = 0.0
    total_prompt_tokens: int = 0
    total_cot_tokens: int = 0
    total_tool_turns: int = 0
    min_prompt_length: int = 0
    max_prompt_length: int = 0
    min_cot_tokens: int = 0
    max_cot_tokens: int = 0
    min_tool_turns: int = 0
    max_tool_turns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProvenanceChain:
    """Cryptographic provenance record linking source ingots, runs, and lot hashes."""

    product_id: str
    product_version: str
    product_hash: str
    product_spec_hash: str
    curation_ledger_hash: str
    source_ingot_ledger_hash: str
    lineage_manifest_hash: str
    rights_manifest_hash: str
    sha256sums_hash: str
    source_ingots_count: int
    source_ingots: list[dict[str, Any]] = field(default_factory=list)
    lineage_events_count: int = 0
    lineage_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionalDataCard:
    """Datasheet for Datasets per Gebru et al. standard with cryptographic grounding."""

    schema_id: str = "goldtrace.mint.datacard.v1"
    product_id: str = ""
    product_version: str = "0.0.0"
    lot_hash: str = ""
    created_at: str = ""
    license: str = "UNSPECIFIED"
    fixture_only: bool = False
    task_taxonomy: list[str] = field(default_factory=list)
    token_statistics: TokenStatistics = field(default_factory=TokenStatistics)
    provenance_chain: ProvenanceChain = field(
        default_factory=lambda: ProvenanceChain(
            product_id="",
            product_version="",
            product_hash="",
            product_spec_hash="",
            curation_ledger_hash="",
            source_ingot_ledger_hash="",
            lineage_manifest_hash="",
            rights_manifest_hash="",
            sha256sums_hash="",
            source_ingots_count=0,
        )
    )
    gebru_sections: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _tokenize_text(text: str) -> int:
    """Deterministic token estimation (whitespace & punctuation split approximation ~4 chars/token or word split)."""
    if not text:
        return 0
    words = len(text.split())
    char_estimate = math.ceil(len(text) / 4)
    return max(words, char_estimate)


def _extract_cot_and_turns(record: dict[str, Any]) -> tuple[int, int, int]:
    """Extract prompt tokens, CoT reasoning tokens, and tool execution turns from a record."""
    prompt_tokens = 0
    cot_tokens = 0
    tool_turns = 0

    if "prompt_tokens" in record:
        prompt_tokens += int(record["prompt_tokens"])
    if "cot_reasoning_tokens" in record or "reasoning_tokens" in record or "cot_tokens" in record:
        cot_tokens += int(record.get("cot_reasoning_tokens") or record.get("reasoning_tokens") or record.get("cot_tokens") or 0)
    if "tool_execution_turns" in record or "tool_turns" in record:
        tool_turns += int(record.get("tool_execution_turns") or record.get("tool_turns") or 0)

    traj = record.get("trajectory_summary") or {}
    if isinstance(traj, dict):
        if "prompt_tokens" in traj and prompt_tokens == 0:
            prompt_tokens += int(traj["prompt_tokens"])
        if ("cot_reasoning_tokens" in traj or "cot_tokens" in traj or "reasoning_tokens" in traj) and cot_tokens == 0:
            cot_tokens += int(traj.get("cot_reasoning_tokens") or traj.get("cot_tokens") or traj.get("reasoning_tokens") or 0)
        if ("tool_execution_turns" in traj or "tool_turns" in traj) and tool_turns == 0:
            tool_turns += int(traj.get("tool_execution_turns") or traj.get("tool_turns") or 0)

    messages = record.get("messages") or record.get("conversation") or record.get("events") or []
    if isinstance(messages, list) and messages:
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = str(msg.get("content", ""))
            
            reasoning = msg.get("reasoning") or msg.get("thought") or msg.get("cot") or ""
            if reasoning:
                cot_tokens += _tokenize_text(str(reasoning))
            else:
                think_matches = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL)
                for t in think_matches:
                    cot_tokens += _tokenize_text(t)

            if role in ("user", "system"):
                prompt_tokens += _tokenize_text(content)

            if role in ("tool", "tool_call", "function") or "tool_calls" in msg or msg.get("event_type") in ("tool_call", "tool_result", "exec"):
                tool_turns += 1

    if prompt_tokens == 0:
        raw_prompt = record.get("prompt") or record.get("input") or record.get("scenario_id") or ""
        if isinstance(raw_prompt, dict):
            raw_prompt = json.dumps(raw_prompt)
        prompt_tokens = _tokenize_text(str(raw_prompt))

    return prompt_tokens, cot_tokens, tool_turns


def compute_token_statistics(records: list[dict[str, Any]]) -> TokenStatistics:
    """Compute token metrics across all records in dataset."""
    if not records:
        return TokenStatistics()

    total_records = len(records)
    total_prompt_tokens = 0
    total_cot_tokens = 0
    total_tool_turns = 0

    prompt_lengths = []
    cot_lengths = []
    tool_counts = []

    for rec in records:
        p_tok, c_tok, t_turns = _extract_cot_and_turns(rec)
        prompt_lengths.append(p_tok)
        cot_lengths.append(c_tok)
        tool_counts.append(t_turns)

        total_prompt_tokens += p_tok
        total_cot_tokens += c_tok
        total_tool_turns += t_turns

    total_tokens = total_prompt_tokens + total_cot_tokens

    return TokenStatistics(
        total_records=total_records,
        total_tokens=total_tokens,
        mean_prompt_length=round(total_prompt_tokens / total_records, 2),
        mean_cot_reasoning_tokens=round(total_cot_tokens / total_records, 2),
        mean_tool_execution_turns=round(total_tool_turns / total_records, 2),
        total_prompt_tokens=total_prompt_tokens,
        total_cot_tokens=total_cot_tokens,
        total_tool_turns=total_tool_turns,
        min_prompt_length=min(prompt_lengths),
        max_prompt_length=max(prompt_lengths),
        min_cot_tokens=min(cot_lengths),
        max_cot_tokens=max(cot_lengths),
        min_tool_turns=min(tool_counts),
        max_tool_turns=max(tool_counts),
    )


def extract_task_taxonomy(product_spec: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    """Extract hierarchical and domain task taxonomy tags from product spec and records."""
    taxonomy_set = set()

    intended = product_spec.get("intended_use") or []
    for item in intended:
        taxonomy_set.add(str(item))

    coverage = product_spec.get("required_coverage") or []
    for item in coverage:
        taxonomy_set.add(f"coverage:{item}")

    modality = (product_spec.get("target_dataset_form") or {}).get("modality")
    if modality:
        taxonomy_set.add(f"modality:{modality}")

    for rec in records:
        scen = rec.get("scenario_id")
        if scen:
            taxonomy_set.add(f"scenario:{scen}")
        labels = rec.get("labels") or {}
        if isinstance(labels, dict):
            for rc in labels.get("reason_codes") or []:
                taxonomy_set.add(f"tag:{rc}")

    return sorted(list(taxonomy_set))


def build_gebru_sections(
    product_spec: dict[str, Any],
    token_stats: TokenStatistics,
    provenance: ProvenanceChain,
    rights: dict[str, Any],
) -> dict[str, Any]:
    """Compile Gebru et al. Datasheets for Datasets standard sections."""
    product_id = product_spec.get("product_id", "unnamed-lot")
    license_str = rights.get("license") or (product_spec.get("rights_requirements") or {}).get("license") or "UNSPECIFIED"

    return {
        "motivation": {
            "purpose": f"Curated institutional training & evaluation corpus for {product_id}.",
            "created_by": "Gold-Trace-Dataworks Automated Refinery & Mint Pipeline",
            "funded_by": "Operator / Gold-Trace-Dataworks Sovereignty Engine",
            "intended_use": product_spec.get("intended_use") or ["Local agent evaluation and post-training"],
            "prohibited_use": product_spec.get("prohibited_use") or ["Unredacted public distribution", "Uncertified commercial sale"],
        },
        "composition": {
            "total_records": token_stats.total_records,
            "total_tokens": token_stats.total_tokens,
            "mean_prompt_length": token_stats.mean_prompt_length,
            "mean_cot_reasoning_tokens": token_stats.mean_cot_reasoning_tokens,
            "mean_tool_execution_turns": token_stats.mean_tool_execution_turns,
            "modality": (product_spec.get("target_dataset_form") or {}).get("modality", "agent_trace"),
            "is_confidential": False,
            "contains_pii_or_secrets": "Strictly audited and redacted via Refinery privacy policy prior to Mint stage.",
        },
        "collection_process": {
            "methodology": "Captured in sealed gVisor runtime sandboxes with cryptographic event logging.",
            "source_ingots_considered": provenance.source_ingots_count,
            "provenance_mechanisms": "Hash-chained event stream verified against SHA-256 bundle manifests.",
            "temporal_coverage": "Discrete agent execution trajectories.",
        },
        "preprocessing_and_cleaning": {
            "refinery_actions": [
                "Bundle digest verification",
                "Normalization to canonical event grammar",
                "Automated PII, secret, and high-entropy key redaction",
                "Ingot casting without semantic judgment fields",
            ],
            "mint_curation": "Rule-based and rubric-guided multi-dimension curation filtering (INCLUDE/EXCLUDE/HOLD).",
        },
        "uses": {
            "suitable_applications": [
                "Supervised Fine-Tuning (SFT) of tool-use and reasoning models",
                "Direct Preference Optimization (DPO) and trajectory correction",
                "Standardized Agent Benchmark Evaluation via EvalFoundry",
            ],
            "unsuitable_applications": [
                "Direct unconstrained internet execution without containment",
                "Training without safety alignment",
            ],
        },
        "distribution": {
            "license": license_str,
            "fixture_only": bool(product_spec.get("fixture_only", False)),
            "rights_status": rights.get("rights_status", "declared"),
            "access_restrictions": "Vault admission gated by Hallmark PASS certification.",
        },
        "maintenance": {
            "custodian": "Institutional Vault Administrator",
            "versioning_policy": "Immutable cryptographic lot hashes; updates minted as new product versions.",
            "auditability": "Full SHA256SUMS and lineage graph included in product manifest.",
        },
    }


def generate_datacard(
    lot_dir: Path,
    *,
    out_dir: Path | None = None,
) -> tuple[InstitutionalDataCard, Path, Path]:
    """Generate DATACARD.md and datacard.json from a built Product Lot."""
    lot_dir = Path(lot_dir).resolve()
    if out_dir is None:
        out_dir = lot_dir / "reports"
    else:
        out_dir = Path(out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = lot_dir / "product-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing product-manifest.json in {lot_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    spec_path = lot_dir / "product-spec.json"
    if spec_path.is_file():
        product_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    else:
        product_spec = {
            "product_id": manifest.get("product_id", "unknown"),
            "product_version": manifest.get("product_version", "0.0.0"),
            "fixture_only": manifest.get("fixture_only", False),
        }

    records: list[dict[str, Any]] = []
    train_path = lot_dir / "dataset" / "train.jsonl"
    if train_path.is_file():
        for line in train_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    val_path = lot_dir / "dataset" / "validation.jsonl"
    if val_path.is_file():
        for line in val_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    source_ingots: list[dict[str, Any]] = []
    src_ingots_path = lot_dir / "provenance" / "source-ingots.jsonl"
    if src_ingots_path.is_file():
        for line in src_ingots_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                source_ingots.append(json.loads(line))

    lineage_events: list[dict[str, Any]] = []
    lineage_path = lot_dir / "provenance" / "lineage.jsonl"
    if lineage_path.is_file():
        for line in lineage_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                lineage_events.append(json.loads(line))

    rights_path = lot_dir / "provenance" / "rights.json"
    rights = json.loads(rights_path.read_text(encoding="utf-8")) if rights_path.is_file() else {}

    sums_path = lot_dir / "SHA256SUMS"
    sums_hash = sha256_file(sums_path) if sums_path.is_file() else ""

    token_stats = compute_token_statistics(records)
    task_taxonomy = extract_task_taxonomy(product_spec, records)

    provenance_chain = ProvenanceChain(
        product_id=manifest.get("product_id", ""),
        product_version=manifest.get("product_version", ""),
        product_hash=manifest.get("product_hash", ""),
        product_spec_hash=manifest.get("product_spec_hash", ""),
        curation_ledger_hash=manifest.get("curation_ledger_hash", ""),
        source_ingot_ledger_hash=manifest.get("source_ingot_ledger_hash", ""),
        lineage_manifest_hash=manifest.get("lineage_manifest_hash", ""),
        rights_manifest_hash=manifest.get("rights_manifest_hash", ""),
        sha256sums_hash=sums_hash,
        source_ingots_count=len(source_ingots),
        source_ingots=source_ingots,
        lineage_events_count=len(lineage_events),
        lineage_events=lineage_events,
    )

    gebru = build_gebru_sections(product_spec, token_stats, provenance_chain, rights)

    datacard = InstitutionalDataCard(
        schema_id="goldtrace.mint.datacard.v1",
        product_id=manifest.get("product_id", ""),
        product_version=manifest.get("product_version", "0.0.0"),
        lot_hash=manifest.get("product_hash", ""),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        license=rights.get("license") or (product_spec.get("rights_requirements") or {}).get("license") or "UNSPECIFIED",
        fixture_only=bool(manifest.get("fixture_only", False)),
        task_taxonomy=task_taxonomy,
        token_statistics=token_stats,
        provenance_chain=provenance_chain,
        gebru_sections=gebru,
    )

    json_path = out_dir / "datacard.json"
    json_path.write_text(json.dumps(datacard.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_path = out_dir / "DATACARD.md"
    md_content = render_datacard_markdown(datacard)
    md_path.write_text(md_content, encoding="utf-8")

    return datacard, json_path, md_path


def render_datacard_markdown(card: InstitutionalDataCard) -> str:
    """Render InstitutionalDataCard into standard Gebru et al. Markdown format."""
    gebru = card.gebru_sections
    stats = card.token_statistics
    prov = card.provenance_chain

    lines = [
        f"# Institutional Datasheet: {card.product_id} (v{card.product_version})",
        "",
        "> **Standards Conformance**: Gebru et al. *Datasheets for Datasets* & Gold-Trace-Dataworks Sovereign Assurance.",
        "",
        "## 1. Executive Summary & Verification Seals",
        "",
        f"- **Product ID**: `{card.product_id}`",
        f"- **Version**: `{card.product_version}`",
        f"- **Canonical Lot Hash**: `{card.lot_hash}`",
        f"- **License Certification**: `{card.license}`",
        f"- **Fixture-Only Mode**: `{card.fixture_only}`",
        f"- **Generated Timestamp**: `{card.created_at}`",
        f"- **Schema**: `{card.schema_id}`",
        "",
        "## 2. Token & Interaction Statistics",
        "",
        "| Metric | Value |",
        "|:-------|:------|",
        f"| **Total Records** | {stats.total_records:,} |",
        f"| **Total Tokens (Prompt + CoT)** | {stats.total_tokens:,} |",
        f"| **Total Prompt Tokens** | {stats.total_prompt_tokens:,} |",
        f"| **Total CoT Reasoning Tokens** | {stats.total_cot_tokens:,} |",
        f"| **Mean Prompt Length** | {stats.mean_prompt_length:.2f} tokens |",
        f"| **Mean CoT Reasoning Tokens** | {stats.mean_cot_reasoning_tokens:.2f} tokens |",
        f"| **Mean Tool Execution Turns** | {stats.mean_tool_execution_turns:.2f} turns |",
        f"| **Prompt Range [Min, Max]** | [{stats.min_prompt_length}, {stats.max_prompt_length}] tokens |",
        f"| **CoT Range [Min, Max]** | [{stats.min_cot_tokens}, {stats.max_cot_tokens}] tokens |",
        f"| **Tool Turns Range [Min, Max]** | [{stats.min_tool_turns}, {stats.max_tool_turns}] turns |",
        "",
        "## 3. Task Taxonomy & Domain Coverage",
        "",
    ]

    if card.task_taxonomy:
        for item in card.task_taxonomy:
            lines.append(f"- `{item}`")
    else:
        lines.append("- *(No specific taxonomy tags recorded)*")

    lines.extend([
        "",
        "## 4. Cryptographic Provenance & Lineage Chain",
        "",
        "| Manifest / Artifact | SHA-256 Digest |",
        "|:--------------------|:---------------|",
        f"| **Product Hash** | `{prov.product_hash}` |",
        f"| **Product Spec** | `{prov.product_spec_hash}` |",
        f"| **Curation Ledger** | `{prov.curation_ledger_hash}` |",
        f"| **Source Ingot Ledger** | `{prov.source_ingot_ledger_hash}` |",
        f"| **Lineage Manifest** | `{prov.lineage_manifest_hash}` |",
        f"| **Rights Manifest** | `{prov.rights_manifest_hash}` |",
        f"| **SHA256SUMS Manifest** | `{prov.sha256sums_hash}` |",
        "",
        f"- **Source Ingot Count**: {prov.source_ingots_count}",
        f"- **Lineage Events Count**: {prov.lineage_events_count}",
        "",
        "## 5. Datasheet Sections (Gebru et al.)",
        "",
        "### 5.1 Motivation",
        f"- **Purpose**: {gebru.get('motivation', {}).get('purpose', '')}",
        f"- **Created By**: {gebru.get('motivation', {}).get('created_by', '')}",
        f"- **Funding**: {gebru.get('motivation', {}).get('funded_by', '')}",
        "- **Intended Use**:",
    ])

    for u in gebru.get("motivation", {}).get("intended_use", []):
        lines.append(f"  - {u}")

    lines.append("- **Prohibited Use**:")
    for p in gebru.get("motivation", {}).get("prohibited_use", []):
        lines.append(f"  - {p}")

    lines.extend([
        "",
        "### 5.2 Composition",
        f"- **Modality**: {gebru.get('composition', {}).get('modality', '')}",
        f"- **Total Records**: {gebru.get('composition', {}).get('total_records', 0)}",
        f"- **Privacy & Confidentiality**: {gebru.get('composition', {}).get('contains_pii_or_secrets', '')}",
        "",
        "### 5.3 Collection Process",
        f"- **Methodology**: {gebru.get('collection_process', {}).get('methodology', '')}",
        f"- **Provenance Grounding**: {gebru.get('collection_process', {}).get('provenance_mechanisms', '')}",
        "",
        "### 5.4 Preprocessing & Cleaning",
        "- **Refinery Invariant Verification**:",
    ])

    for step in gebru.get("preprocessing_and_cleaning", {}).get("refinery_actions", []):
        lines.append(f"  - {step}")

    lines.append(f"- **Mint Curation**: {gebru.get('preprocessing_and_cleaning', {}).get('mint_curation', '')}")

    lines.extend([
        "",
        "### 5.5 Uses & Applications",
        "- **Suitable Applications**:",
    ])
    for app in gebru.get("uses", {}).get("suitable_applications", []):
        lines.append(f"  - {app}")

    lines.append("- **Unsuitable Applications**:")
    for unapp in gebru.get("uses", {}).get("unsuitable_applications", []):
        lines.append(f"  - {unapp}")

    lines.extend([
        "",
        "### 5.6 Distribution & Governance",
        f"- **License**: `{gebru.get('distribution', {}).get('license', '')}`",
        f"- **Rights Status**: `{gebru.get('distribution', {}).get('rights_status', '')}`",
        f"- **Access Gate**: {gebru.get('distribution', {}).get('access_restrictions', '')}",
        "",
        "### 5.7 Maintenance & Custodianship",
        f"- **Custodian**: {gebru.get('maintenance', {}).get('custodian', '')}",
        f"- **Versioning Policy**: {gebru.get('maintenance', {}).get('versioning_policy', '')}",
        f"- **Auditability**: {gebru.get('maintenance', {}).get('auditability', '')}",
        "",
    ])

    return "\n".join(lines)
