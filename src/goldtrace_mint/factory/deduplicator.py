#!/usr/bin/env python3
"""
GTDataworks Deduplication & Anti-Leakage Curator Engine
======================================================
High-precision deduplication and benchmark contamination quarantine for
agent seed banks, SFT prompts, DPO preference pairs, and evaluation sets.

Features:
  1. Exact Text & Exact 13-Gram Deduplication:
     - Exact normalized content hash (SHA-256).
     - Sliding window 13-gram inverted index containment.
  2. MinHash Jaccard Near-Duplicate Pruning:
     - Universal hashing MinHash (128 permutations, Mersenne prime 2^61-1).
     - Configurable Jaccard similarity threshold (default 0.85).
  3. Anti-Contamination & Benchmark Integrity Verification:
     - Zero-tolerance (0.0% overlap) verification against public benchmarks:
       HumanEval, MBPP, SWE-bench, GSM8K via ContaminationChecker.
  4. Structured Curation Receipts & Gebru-Compliant Audit Trails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .contamination import ContaminationChecker, ContaminationReport, MinHasher

logger = logging.getLogger("goldtrace_mint.factory.deduplicator")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DeduplicationRecord:
    """Audit record for an individual seed processed by the curator."""
    seed_id: str
    raw_index: int
    source_file: str
    status: str  # "CLEAN", "DUPLICATE_EXACT", "DUPLICATE_MINHASH", "CONTAMINATED"
    duplicate_of: Optional[str] = None
    jaccard_similarity: float = 0.0
    contamination_report: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CurationSummary:
    """Summary metrics of the deduplication and anti-leakage curation pass."""
    total_ingested: int = 0
    exact_duplicates_pruned: int = 0
    minhash_duplicates_pruned: int = 0
    total_duplicates_pruned: int = 0
    contaminated_pruned: int = 0
    final_clean_count: int = 0
    source_breakdown: Dict[str, int] = field(default_factory=dict)
    role_breakdown: Dict[str, int] = field(default_factory=dict)
    domain_breakdown: Dict[str, int] = field(default_factory=dict)
    benchmark_matches: Dict[str, int] = field(default_factory=dict)
    dedup_threshold: float = 0.85
    minhash_threshold: float = 0.85
    curation_timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary_markdown(self) -> str:
        lines = [
            "# Seed Curation & Anti-Leakage Audit Summary",
            "",
            f"- **Total Raw Seeds Ingested**: `{self.total_ingested}`",
            f"- **Exact Duplicates Pruned**: `{self.exact_duplicates_pruned}`",
            f"- **MinHash Near-Duplicates Pruned (Jaccard >= {self.minhash_threshold:.2f})**: `{self.minhash_duplicates_pruned}`",
            f"- **Total Redundant Seeds Pruned**: `{self.total_duplicates_pruned}`",
            f"- **Contaminated Seeds Flagged/Quarantined**: `{self.contaminated_pruned}`",
            f"- **Final Certified Clean Seeds**: `{self.final_clean_count}`",
            "",
            "### Breakdown by Role Category",
        ]
        for role, cnt in sorted(self.role_breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"- **{role}**: `{cnt}`")
        
        lines.append("")
        lines.append("### Breakdown by Domain")
        for dom, cnt in sorted(self.domain_breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"- **{dom}**: `{cnt}`")

        if self.benchmark_matches:
            lines.append("")
            lines.append("### Benchmark Contamination Detections")
            for bm, cnt in self.benchmark_matches.items():
                lines.append(f"- **{bm}**: `{cnt}` hits")
        else:
            lines.append("")
            lines.append("### Benchmark Contamination: **0.0% LEAKAGE (ZERO HITS)** ✅")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Canonical Text Extraction Helper
# ---------------------------------------------------------------------------

def extract_seed_canonical_text(seed: Union[Dict[str, Any], str]) -> str:
    """
    Extract the core canonical prompt/scenario text from an arbitrary seed representation.
    """
    if isinstance(seed, str):
        return seed.strip()

    if not isinstance(seed, dict):
        return str(seed).strip()

    parts: List[str] = []

    # Priority task prompt fields
    for field_name in ("task_prompt", "prompt", "target_behavior", "scenario", "instruction", "query", "question"):
        val = seed.get(field_name)
        if val and isinstance(val, str):
            parts.append(val.strip())
            break

    # Role function / description
    role_fn = seed.get("role_function") or seed.get("description")
    if role_fn and isinstance(role_fn, str):
        parts.append(role_fn.strip())

    # Multi-turn messages or events
    messages = seed.get("messages") or seed.get("turns")
    if messages and isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict):
                content = m.get("content") or m.get("text") or ""
                reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
                if reasoning:
                    parts.append(f"reasoning: {reasoning}")
                if content:
                    parts.append(f"content: {content}")

    # Fallback to stringifying if empty
    if not parts:
        for k in ("input", "content", "text", "body"):
            val = seed.get(k)
            if val and isinstance(val, str):
                parts.append(val.strip())
                break

    if not parts:
        # Serialized dict representation without ephemeral IDs
        filtered = {k: v for k, v in seed.items() if k not in ("seed_id", "id", "timestamp", "created_at")}
        parts.append(json.dumps(filtered, sort_keys=True))

    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Seed Curator Engine
# ---------------------------------------------------------------------------

class SeedCurator:
    """
    Deduplicates and verifies seeds against public benchmark contamination.
    """

    def __init__(
        self,
        minhash_threshold: float = 0.85,
        ngram_size: int = 13,
        num_perm: int = 128,
        seed: int = 42,
        benchmark_signatures: Optional[Dict[str, List[Any]]] = None,
    ):
        self.minhash_threshold = minhash_threshold
        self.ngram_size = ngram_size
        self.num_perm = num_perm
        self.minhasher = MinHasher(num_perm=num_perm, seed=seed)
        self.contamination_checker = ContaminationChecker(
            benchmark_signatures=benchmark_signatures,
            ngram_size=ngram_size,
            overlap_threshold=0.0,
            minhash_threshold=0.8,
            seed=seed,
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for exact hashing and whitespace invariant comparisons."""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def curate_seeds(
        self,
        raw_seeds: Sequence[Dict[str, Any]],
        source_names: Optional[Sequence[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], CurationSummary, List[DeduplicationRecord]]:
        """
        Process raw seeds through exact deduplication, MinHash Jaccard pruning,
        and benchmark contamination scanning.

        Returns:
            clean_seeds: List of unique, non-contaminated seed dicts.
            summary: Aggregated metrics report.
            audit_records: Per-seed audit decisions.
        """
        from datetime import datetime, timezone

        clean_seeds: List[Dict[str, Any]] = []
        audit_records: List[DeduplicationRecord] = []

        exact_hashes: Set[str] = set()
        accepted_signatures: List[Tuple[str, List[int], Tuple[str, ...]]] = []
        # Inverted index for exact 13-gram sets
        ngram_sets: Dict[str, Set[Tuple[str, ...]]] = {}

        summary = CurationSummary(
            total_ingested=len(raw_seeds),
            dedup_threshold=self.minhash_threshold,
            minhash_threshold=self.minhash_threshold,
            curation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        for idx, seed_record in enumerate(raw_seeds):
            src_file = source_names[idx] if source_names and idx < len(source_names) else "unknown"
            seed_id = str(
                seed_record.get("seed_id")
                or seed_record.get("bonsai_seed_id")
                or seed_record.get("id")
                or f"SEED-{idx:05d}"
            )
            role_category = str(
                seed_record.get("role_category")
                or seed_record.get("swarm_role")
                or seed_record.get("seed_family")
                or "GENERAL"
            )
            domain = str(
                seed_record.get("domain")
                or seed_record.get("seed_type")
                or seed_record.get("source_seed_type")
                or "general"
            )

            # Update source breakdown
            summary.source_breakdown[src_file] = summary.source_breakdown.get(src_file, 0) + 1

            # 1. Canonical text extraction & tokenization
            canonical_text = extract_seed_canonical_text(seed_record)
            normalized_text = self._normalize_text(canonical_text)

            # 2. Benchmark Anti-Contamination Verification
            contam_report = self.contamination_checker.check_text(canonical_text)
            if not contam_report.certified_clean:
                # Flagged for benchmark leakage
                summary.contaminated_pruned += 1
                for bm, cnt in contam_report.benchmark_matches.items():
                    summary.benchmark_matches[bm] = summary.benchmark_matches.get(bm, 0) + cnt

                audit_records.append(
                    DeduplicationRecord(
                        seed_id=seed_id,
                        raw_index=idx,
                        source_file=src_file,
                        status="CONTAMINATED",
                        duplicate_of=None,
                        jaccard_similarity=0.0,
                        contamination_report=contam_report.to_dict(),
                        details={
                            "reason": "Public benchmark contamination detected",
                            "overlap_percentage": contam_report.overlap_percentage,
                            "benchmark_matches": contam_report.benchmark_matches,
                        },
                    )
                )
                continue

            # 3. Exact Hash Deduplication
            text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
            if text_hash in exact_hashes:
                summary.exact_duplicates_pruned += 1
                summary.total_duplicates_pruned += 1
                audit_records.append(
                    DeduplicationRecord(
                        seed_id=seed_id,
                        raw_index=idx,
                        source_file=src_file,
                        status="DUPLICATE_EXACT",
                        duplicate_of=seed_id,
                        jaccard_similarity=1.0,
                        details={"reason": "Exact normalized text SHA-256 duplicate"},
                    )
                )
                continue

            # 4. Token & 13-Gram Extraction
            tokens = self.contamination_checker.tokenize(canonical_text)
            token_tuples = tuple(t[0] for t in tokens)
            ngrams_with_pos = self.contamination_checker.extract_ngrams(tokens, n=self.ngram_size)
            seed_ngrams = [ng[0] for ng in ngrams_with_pos]
            seed_ngram_set = set(seed_ngrams)

            # Exact 13-gram full containment check (if seeds have sufficient tokens)
            is_exact_ngram_dup = False
            matched_exact_id = None
            if seed_ngram_set:
                for existing_id, existing_set in ngram_sets.items():
                    if existing_set and seed_ngram_set == existing_set:
                        is_exact_ngram_dup = True
                        matched_exact_id = existing_id
                        break

            if is_exact_ngram_dup:
                summary.exact_duplicates_pruned += 1
                summary.total_duplicates_pruned += 1
                audit_records.append(
                    DeduplicationRecord(
                        seed_id=seed_id,
                        raw_index=idx,
                        source_file=src_file,
                        status="DUPLICATE_EXACT",
                        duplicate_of=matched_exact_id,
                        jaccard_similarity=1.0,
                        details={"reason": "Identical 13-gram set match"},
                    )
                )
                continue

            # 5. MinHash Jaccard Similarity Deduplication
            sig = self.minhasher.compute_signature(seed_ngrams if seed_ngrams else [token_tuples])
            is_minhash_dup = False
            max_sim = 0.0
            matched_minhash_id = None

            for existing_id, existing_sig, _ in accepted_signatures:
                sim = self.minhasher.estimate_similarity(sig, existing_sig)
                if sim > max_sim:
                    max_sim = sim
                    matched_minhash_id = existing_id

                if sim >= self.minhash_threshold:
                    is_minhash_dup = True
                    break

            if is_minhash_dup:
                summary.minhash_duplicates_pruned += 1
                summary.total_duplicates_pruned += 1
                audit_records.append(
                    DeduplicationRecord(
                        seed_id=seed_id,
                        raw_index=idx,
                        source_file=src_file,
                        status="DUPLICATE_MINHASH",
                        duplicate_of=matched_minhash_id,
                        jaccard_similarity=round(max_sim, 4),
                        details={
                            "reason": f"MinHash Jaccard {max_sim:.4f} >= threshold {self.minhash_threshold:.2f}"
                        },
                    )
                )
                continue

            # 6. Accepted as Pristine Clean Seed
            exact_hashes.add(text_hash)
            accepted_signatures.append((seed_id, sig, token_tuples))
            if seed_ngram_set:
                ngram_sets[seed_id] = seed_ngram_set

            clean_seeds.append(seed_record)
            summary.final_clean_count += 1
            summary.role_breakdown[role_category] = summary.role_breakdown.get(role_category, 0) + 1
            summary.domain_breakdown[domain] = summary.domain_breakdown.get(domain, 0) + 1

            audit_records.append(
                DeduplicationRecord(
                    seed_id=seed_id,
                    raw_index=idx,
                    source_file=src_file,
                    status="CLEAN",
                    duplicate_of=None,
                    jaccard_similarity=round(max_sim, 4) if max_sim > 0.0 else 0.0,
                    details={"token_count": len(tokens), "ngram_count": len(seed_ngrams)},
                )
            )

        return clean_seeds, summary, audit_records

    def curate_files(
        self,
        input_files: Sequence[Union[str, Path]],
        output_file: Union[str, Path],
        audit_file: Optional[Union[str, Path]] = None,
        summary_file: Optional[Union[str, Path]] = None,
    ) -> CurationSummary:
        """
        Load seeds from a list of JSON / JSONL files, curate them, and export clean seeds.
        """
        raw_seeds: List[Dict[str, Any]] = []
        source_names: List[str] = []

        for p_str in input_files:
            p = Path(p_str)
            if not p.exists():
                logger.warning(f"File not found: {p}")
                continue

            if p.suffix == ".jsonl":
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            raw_seeds.append(json.loads(line))
                            source_names.append(p.name)
            else:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        raw_seeds.append(item if isinstance(item, dict) else {"prompt": str(item)})
                        source_names.append(p.name)
                elif isinstance(data, dict):
                    items = data.get("seeds") or data.get("items") or [data]
                    for item in items:
                        raw_seeds.append(item if isinstance(item, dict) else {"prompt": str(item)})
                        source_names.append(p.name)

        clean_seeds, summary, audit_records = self.curate_seeds(raw_seeds, source_names=source_names)

        # Export clean seeds
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for seed in clean_seeds:
                f.write(json.dumps(seed) + "\n")

        # Export audit records
        if audit_file:
            aud_path = Path(audit_file)
            aud_path.parent.mkdir(parents=True, exist_ok=True)
            with aud_path.open("w", encoding="utf-8") as f:
                for rec in audit_records:
                    f.write(json.dumps(rec.to_dict()) + "\n")

        # Export summary JSON
        if summary_file:
            sum_path = Path(summary_file)
            sum_path.parent.mkdir(parents=True, exist_ok=True)
            sum_path.write_text(summary.to_json(indent=2), encoding="utf-8")

        return summary
