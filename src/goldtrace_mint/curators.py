from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import sha256_file, sha256_json


class Curator(ABC):
    @abstractmethod
    def decide(self, ingot: dict[str, Any], product_spec: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class ManualFileCurator(Curator):
    """File-based curator for operator decisions and fixtures."""

    def __init__(self, decisions_path: Path):
        self.decisions_path = Path(decisions_path)
        raw = json.loads(self.decisions_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            self._by_ingot = {row["ingot_id"]: row for row in raw}
        elif isinstance(raw, dict) and "decisions" in raw:
            self._by_ingot = {row["ingot_id"]: row for row in raw["decisions"]}
        else:
            self._by_ingot = raw

    def decide(self, ingot: dict[str, Any], product_spec: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
        ingot_id = ingot["ingot_id"]
        if ingot_id not in self._by_ingot:
            raise ValueError(f"unaccounted_for_ingot: {ingot_id}")
        row = self._by_ingot[ingot_id]
        decision = row.get("decision")
        if decision not in {"INCLUDE", "EXCLUDE", "HOLD"}:
            raise ValueError(f"invalid decision for {ingot_id}")
        product_spec_hash = sha256_json(product_spec)
        rubric_hash = sha256_json(rubric)
        ingot_hash = sha256_json(ingot)
        body = {
            "schema_id": "goldtrace.mint.curation_decision.v1",
            "decision_id": row.get("decision_id") or f"dec-{ingot_id}",
            "product_spec_hash": product_spec_hash,
            "ingot_id": ingot_id,
            "ingot_hash": ingot_hash,
            "rubric_id": rubric.get("rubric_id", "unknown"),
            "rubric_version": rubric.get("rubric_version", "0"),
            "rubric_hash": rubric_hash,
            "reviewer_type": "file",
            "reviewer_identity": row.get("reviewer_identity", str(self.decisions_path)),
            "prompt_config_hash": None,
            "decision": decision,
            "reason_codes": list(row.get("reason_codes") or ["manual"]),
            "evidence_refs": list(row.get("evidence_refs") or []),
            "quality_dimensions": dict(row.get("quality_dimensions") or {}),
            "confidence": float(row.get("confidence", 1.0)),
            "proposed_dataset_role": row.get("proposed_dataset_role", "train"),
            "proposed_eval_eligibility": bool(row.get("proposed_eval_eligibility", False)),
            "split_group_id": row.get("split_group_id") or ingot.get("mine_run_id") or ingot_id,
            "reviewed_at": row.get("reviewed_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fixture_only": bool(row.get("fixture_only", product_spec.get("fixture_only", False))),
        }
        body["receipt_hash"] = sha256_json(body)
        return body


class OpenAICompatibleCurator(Curator):
    """Fail-closed scaffold — does not contact network by default."""

    def __init__(self, endpoint: str | None = None, api_key: str | None = None):
        self.endpoint = endpoint
        self.api_key = api_key

    def decide(self, ingot: dict[str, Any], product_spec: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(
            "OpenAICompatibleCurator is fail-closed: configure explicitly and call only with "
            "GOLDTRACE_ALLOW_NETWORK=1; default Mint path uses ManualFileCurator."
        )
