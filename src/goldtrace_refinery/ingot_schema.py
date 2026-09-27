from __future__ import annotations

import json
from typing import Any

from .paths import find_contract

FORBIDDEN_SEMANTIC_FIELDS = frozenset(
    {
        "training_value",
        "commercial_value",
        "reasoning_quality",
        "gold_tier",
        "topic_quality",
        "dataset_include",
    }
)


def load_ingot_schema() -> dict[str, Any]:
    return json.loads(find_contract("ingot.v1.schema.json").read_text(encoding="utf-8"))



def validate_ingot(ingot: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in FORBIDDEN_SEMANTIC_FIELDS:
        if field in ingot:
            errors.append(f"semantic field forbidden in refinery ingot: {field}")
    required = [
        "schema_id",
        "ingot_id",
        "source_class",
        "source_bundle_hash",
        "source_manifest_hash",
        # Present on every ingot so the path that produced it is never implicit.
        "evaluation_path",
        "foundry_evaluation",
        "mechanical_run_status",
        "seal_status",
        "normalized_event_stream_hash",
        "deterministic_results",
        "redaction",
        "quarantine",
        "dedup",
        "lineage_parents",
        "refinery_receipt_hash",
        "refinery_status",
    ]
    for key in required:
        if key not in ingot:
            errors.append(f"missing required field: {key}")
    if ingot.get("schema_id") != "goldtrace.refinery.ingot.v1":
        errors.append("invalid schema_id")
    if ingot.get("source_class") not in {
        "goldtrace_mine_bundle",
        "rune_capture",
        "chat_export",
        "legacy_corpus",
    }:
        errors.append("invalid source_class")
    # jsonschema is a declared runtime dependency. Report an unavailable
    # validator distinctly from a schema violation so a broken install is never
    # mistaken for invalid data — both still fail closed.
    try:
        import jsonschema
    except ImportError as exc:
        errors.append(f"schema validator unavailable (install jsonschema): {exc}")
    else:
        try:
            jsonschema.Draft202012Validator(load_ingot_schema()).validate(ingot)
        except Exception as exc:
            errors.append(f"schema: {exc}")
    return errors
