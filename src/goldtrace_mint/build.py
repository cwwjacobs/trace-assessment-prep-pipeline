from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .content import extract_content
from .curators import Curator, ManualFileCurator
from .hashing import sha256_bytes, sha256_file, sha256_json, sha256_product_lot


FORBIDDEN_CROSS = ("train", "eval")
RECORD_MAP_SCHEMA_ID = "goldtrace.mint.record_map.v1"
RECIPE_TOP_LEVEL_FIELDS: dict[str, tuple[str, ...]] = {
    "sft-chat-v1": ("messages",),
    "tool-use-v1": ("messages",),
    "dpo-preference-v1": ("prompt", "chosen", "rejected"),
    "recovery-v1": ("task", "attempts", "verification"),
    "eval-pairwise-v1": ("id", "task", "input", "split_group_id"),
}


def _write_json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return sha256_file(path)


def _jsonl_line(row: dict[str, Any]) -> bytes:
    return (
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _contract(name: str) -> dict[str, Any]:
    from goldtrace_refinery.paths import find_contract

    return json.loads(find_contract(name).read_text(encoding="utf-8"))


#: Environment variable naming the directory of sealed Labyrinth run bundles.
RUNS_DIR_ENV_VAR = "GOLDTRACE_RUNS_DIR"


def _discover_runs_dir() -> Path:
    """Locate sealed run bundles without assuming an operator machine layout.

    Order: explicit environment variable, the Labyrinth component inside a
    GTDataworks workspace, then the legacy ``mine/runtime/runs`` sibling layout.
    Callers may always pass ``runs_dir`` explicitly to bypass discovery.
    """
    import os

    override = os.environ.get(RUNS_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / ".gitmodules").is_file() and (ancestor / "tools").is_dir():
            for child in sorted(ancestor.iterdir()):
                if child.is_dir() and "labyrinth" in child.name.lower():
                    candidate = child / "runtime" / "runs"
                    if candidate.is_dir():
                        return candidate
            break

    for candidate in (
        Path("mine") / "runtime" / "runs",
        here.parents[2] / "mine" / "runtime" / "runs",
        here.parents[3] / "mine" / "runtime" / "runs",
    ):
        if candidate.is_dir():
            return candidate
    return Path("mine") / "runtime" / "runs"



def _recipe_id(product_spec: dict[str, Any]) -> str:
    explicit = product_spec.get("recipe_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    target = product_spec.get("target_dataset_form")
    record_format = target.get("record_format") if isinstance(target, dict) else None
    if record_format in RECIPE_TOP_LEVEL_FIELDS:
        return str(record_format)
    return "sft-chat-v1"


def _recipe_version(recipe_id: str) -> str:
    suffix = recipe_id.rsplit("-", 1)[-1]
    return suffix if suffix.startswith("v") else "v1"


def _record_format(recipe_id: str) -> str:
    if recipe_id == "tool-use-v1":
        return "openai_chat_tools_jsonl.v1"
    if recipe_id == "sft-chat-v1":
        return "openai_chat_messages_jsonl.v1"
    return f"goldtrace_{recipe_id.replace('-', '_')}_jsonl.v1"


def _model_row(content: dict[str, Any], recipe_id: str) -> dict[str, Any]:
    allowed = RECIPE_TOP_LEVEL_FIELDS.get(recipe_id)
    if allowed is None:
        raise ValueError(f"unsupported Mint recipe: {recipe_id}")
    row = {field: content[field] for field in allowed if field in content}
    if set(row) != set(allowed):
        missing = sorted(set(allowed) - set(row))
        raise ValueError(f"recipe-valid content is missing model fields: {missing}")
    if recipe_id in {"sft-chat-v1", "tool-use-v1"} and not row.get("messages"):
        raise ValueError("chat recipe produced no messages")
    return row


def _tool_use_schema(recipe_id: str) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "goldtrace.mint.tool-use-v1.model-row.v1",
        "title": "OpenAI-style chat tool-use training row",
        "x-goldtrace-recipe-id": recipe_id,
        "type": "object",
        "additionalProperties": False,
        "required": ["messages"],
        "properties": {
            "messages": {
                "type": "array",
                "minItems": 4,
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["role", "content"],
                            "properties": {
                                "role": {"enum": ["system", "user"]},
                                "content": {"type": "string"},
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["role", "content"],
                            "properties": {
                                "role": {"const": "assistant"},
                                "content": {"type": "string"},
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["role", "content", "tool_calls"],
                            "properties": {
                                "role": {"const": "assistant"},
                                "content": {"type": "null"},
                                "tool_calls": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["id", "type", "function"],
                                        "properties": {
                                            "id": {
                                                "type": "string",
                                                "pattern": "^call_[0-9]{4}$",
                                            },
                                            "type": {"const": "function"},
                                            "function": {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "required": ["name", "arguments"],
                                                "properties": {
                                                    "name": {
                                                        "type": "string",
                                                        "minLength": 1,
                                                    },
                                                    "arguments": {"type": "string"},
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["role", "content", "tool_call_id"],
                            "properties": {
                                "role": {"const": "tool"},
                                "content": {"type": "string"},
                                "tool_call_id": {
                                    "type": "string",
                                    "pattern": "^call_[0-9]{4}$",
                                },
                            },
                        },
                    ]
                },
            }
        },
    }


def _dataset_schema(recipe_id: str) -> dict[str, Any]:
    if recipe_id == "tool-use-v1":
        return _tool_use_schema(recipe_id)
    fields = RECIPE_TOP_LEVEL_FIELDS[recipe_id]
    properties: dict[str, Any] = {field: {} for field in fields}
    if "messages" in properties:
        properties["messages"] = {"type": "array", "minItems": 1}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"goldtrace.mint.{recipe_id}.model-row.v1",
        "x-goldtrace-recipe-id": recipe_id,
        "type": "object",
        "additionalProperties": False,
        "required": list(fields),
        "properties": properties,
    }


def _record_map_row(
    *,
    content: dict[str, Any],
    model_row: dict[str, Any],
    recipe_id: str,
    record_id: str,
    split: str,
    row_index: int,
    ingot: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    provenance = content.get("_provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    allowed = set(RECIPE_TOP_LEVEL_FIELDS[recipe_id]) | {"_provenance"}
    explicit_provenance_keys = {
        "session_id",
        "turn_id",
        "user_input_event_ids",
        "agent_message_event_ids",
        "request_event_ids",
        "result_event_ids",
        "effect_event_ids",
        "source_interactive_evidence_sha256",
        "source_index_sha256",
        "normalized_tool_calls",
    }
    extractor_metadata = {
        key: value for key, value in content.items() if key not in allowed
    }
    extractor_metadata.update(
        {
            key: value
            for key, value in provenance.items()
            if key not in explicit_provenance_keys
        }
    )
    result = {
        "schema_id": RECORD_MAP_SCHEMA_ID,
        "record_id": record_id,
        "recipe_id": recipe_id,
        "split": split,
        "dataset_path": f"dataset/{'train' if split == 'train' else 'validation'}.jsonl",
        "row_index": row_index,
        "rendered_row_sha256": sha256_bytes(_jsonl_line(model_row)),
        "source_ingot_id": decision["ingot_id"],
        "source_ingot_hash": decision["ingot_hash"],
        "source_bundle_hash": ingot["source_bundle_hash"],
        "mine_run_id": ingot.get("mine_run_id"),
        "split_group_id": decision["split_group_id"],
        "curation_decision_id": decision["decision_id"],
        "session_id": provenance.get("session_id"),
        "turn_id": provenance.get("turn_id"),
        "user_input_event_ids": list(provenance.get("user_input_event_ids") or []),
        "agent_message_event_ids": list(
            provenance.get("agent_message_event_ids") or []
        ),
        "request_event_ids": list(provenance.get("request_event_ids") or []),
        "result_event_ids": list(provenance.get("result_event_ids") or []),
        "effect_event_ids": list(provenance.get("effect_event_ids") or []),
        "source_interactive_evidence_sha256": provenance.get(
            "source_interactive_evidence_sha256"
        ),
        "source_index_sha256": provenance.get("source_index_sha256"),
        "normalized_tool_calls": list(
            provenance.get("normalized_tool_calls") or []
        ),
        "extractor_metadata": extractor_metadata,
    }
    try:
        schema = _contract("mint-record-map.v1.schema.json")
        errors = sorted(
            Draft202012Validator(schema).iter_errors(result),
            key=lambda error: list(error.path),
        )
        if errors:
            raise ValueError(f"Mint provenance record map invalid: {errors[0].message}")
    except FileNotFoundError:
        pass
    return result



def load_ingots(ingot_paths: list[Path]) -> list[dict[str, Any]]:
    ingots = []
    for path in ingot_paths:
        source = Path(path)
        if source.is_dir():
            source = source / "ingot.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        if "ingot_id" not in data:
            raise ValueError(f"ingot lacks ingot_id: {source}")
        ingots.append(data)
    return ingots


def _ingot_root(path: Path) -> Path:
    source = Path(path)
    return source.resolve() if source.is_dir() else source.resolve().parent


def build_product_lot(
    *,
    ingot_paths: list[Path],
    product_spec: dict[str, Any],
    rubric: dict[str, Any],
    curator: Curator,
    out_dir: Path,
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing product lot: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if runs_dir is None:
        runs_dir = _discover_runs_dir()
    runs_dir = Path(runs_dir).resolve()


    recipe_id = _recipe_id(product_spec)

    ingots = load_ingots(ingot_paths)
    ingot_roots = {
        ingot["ingot_id"]: _ingot_root(path)
        for ingot, path in zip(ingots, ingot_paths, strict=True)
    }
    if len(ingot_roots) != len(ingots):
        raise ValueError("duplicate ingot_id in Mint input")
    decisions: list[dict[str, Any]] = []
    for ingot in ingots:
        decisions.append(curator.decide(ingot, product_spec, rubric))

    considered = {i["ingot_id"] for i in ingots}
    accounted = {d["ingot_id"] for d in decisions}
    if considered != accounted:
        missing = considered - accounted
        raise ValueError(f"curation run with unaccounted-for ingot(s): {sorted(missing)}")

    included = [d for d in decisions if d["decision"] == "INCLUDE"]
    excluded = [d for d in decisions if d["decision"] == "EXCLUDE"]
    held = [d for d in decisions if d["decision"] == "HOLD"]
    if len(included) + len(excluded) + len(held) != len(decisions):
        raise ValueError("decision reconciliation failed")

    # Group-aware split: same split_group_id cannot appear in both train and eval
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    eval_cases: list[dict[str, Any]] = []
    train_groups: set[str] = set()
    eval_groups: set[str] = set()
    ingot_by_id = {i["ingot_id"]: i for i in ingots}
    training_record_count = 0
    record_maps: list[dict[str, Any]] = []

    for decision in included:
        ingot = ingot_by_id[decision["ingot_id"]]
        group = decision["split_group_id"]
        role = decision.get("proposed_dataset_role") or "train"

        # Extract actual training content from the ingot
        training_records = extract_content(
            ingot,
            recipe_id,
            runs_dir,
            ingot_root=ingot_roots[ingot["ingot_id"]],
        )

        if not training_records:
            raise ValueError(
                "included ingot produced no recipe-valid model-facing content: "
                f"{decision['ingot_id']} ({recipe_id})"
            )

        for content in training_records:
            record_id = f"rec-{decision['ingot_id']}-{training_record_count:04d}"
            training_record_count += 1
            model_row = _model_row(content, recipe_id)

            if role == "eval" or decision.get("proposed_eval_eligibility"):
                if group in train_groups:
                    raise ValueError(f"leakage_group_cross_split: {group}")
                eval_groups.add(group)
                eval_cases.append({
                    "id": f"eval-{record_id}",
                    "source_id": decision["ingot_id"],
                    "task": content.get("task", "Evaluate model response"),
                    "input": {
                        "scenario_id": ingot.get("scenario_id"),
                        "seal_status": ingot.get("seal_status"),
                        "messages": model_row.get("messages", []),
                    },
                    "split_group_id": group,
                    "label": {
                        "expected_decision": "INCLUDE",
                        "reason_codes": decision["reason_codes"],
                    },
                })
            elif role == "validation":
                if group in eval_groups:
                    raise ValueError(f"leakage_group_cross_split: {group}")
                train_groups.add(group)
                row_index = len(val_rows)
                val_rows.append(model_row)
                record_maps.append(
                    _record_map_row(
                        content=content,
                        model_row=model_row,
                        recipe_id=recipe_id,
                        record_id=record_id,
                        split="validation",
                        row_index=row_index,
                        ingot=ingot,
                        decision=decision,
                    )
                )
            else:
                if group in eval_groups:
                    raise ValueError(f"leakage_group_cross_split: {group}")
                train_groups.add(group)
                row_index = len(train_rows)
                train_rows.append(model_row)
                record_maps.append(
                    _record_map_row(
                        content=content,
                        model_row=model_row,
                        recipe_id=recipe_id,
                        record_id=record_id,
                        split="train",
                        row_index=row_index,
                        ingot=ingot,
                        decision=decision,
                    )
                )

    # Matching eval pack is required even if empty cases — create minimal self-check pack
    if not eval_cases:
        # Build a non-train eval task from product-spec coverage, not a train copy
        eval_cases.append(
            {
                "id": "eval-fixture-coverage-001",
                "source_id": "product_spec",
                "task": "product_spec_coverage_check",
                "input": {
                    "product_id": product_spec.get("product_id"),
                    "required_coverage": product_spec.get("required_coverage") or [],
                },
                "split_group_id": f"eval-spec-{product_spec.get('product_id')}",
                "label": {"expected": "coverage_documented"},
            }
        )

    overlap = train_groups & eval_groups
    if overlap:
        raise ValueError(f"leakage_group_cross_split: {sorted(overlap)}")

    dataset_version = product_spec.get("product_version", "0.0.0")
    eval_version = product_spec.get("product_version", "0.0.0")

    # Layout
    dataset_dir = out_dir / "dataset"
    eval_dir = out_dir / "eval"
    curation_dir = out_dir / "curation"
    provenance_dir = out_dir / "provenance"
    reports_dir = out_dir / "reports"
    trainer_dir = out_dir / "trainer-upload"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_hash = _write_jsonl(dataset_dir / "train.jsonl", train_rows)
    val_hash = _write_jsonl(dataset_dir / "validation.jsonl", val_rows)
    schema = _dataset_schema(recipe_id)
    schema_hash = _write_json(dataset_dir / "schema.json", schema)
    record_map_hash = _write_jsonl(
        provenance_dir / "record-map.jsonl", record_maps
    )

    trainer_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dataset_dir / "train.jsonl", trainer_dir / "train.jsonl")
    shutil.copyfile(
        dataset_dir / "validation.jsonl", trainer_dir / "validation.jsonl"
    )
    shutil.copyfile(dataset_dir / "schema.json", trainer_dir / "schema.json")
    example_rows = train_rows or val_rows
    (trainer_dir / "EXAMPLE.jsonl").write_bytes(
        _jsonl_line(example_rows[0]) if example_rows else b""
    )
    record_format = _record_format(recipe_id)
    (trainer_dir / "README.md").write_text(
        "# Trainer Upload\n\n"
        f"Format: `{record_format}` for recipe `{recipe_id}`.\n\n"
        "- Upload or use `train.jsonl` as the training data.\n"
        "- Optionally use `validation.jsonl` as validation data.\n"
        "- `schema.json` describes the exact model-visible JSON object.\n"
        "- `EXAMPLE.jsonl` is for inspection only; do not upload it as an "
        "additional training split.\n"
        "- Do **not** feed `provenance/`, `curation/`, `reports/`, manifests, "
        "receipts, or the complete product ZIP to a model.\n\n"
        "This directory states one exact JSONL format; it does not claim "
        "compatibility with every fine-tuning service.\n",
        encoding="utf-8",
    )
    trainer_manifest = {
        "schema_id": "goldtrace.mint.trainer_upload_manifest.v1",
        "record_format": record_format,
        "recipe_id": recipe_id,
        "recipe_version": _recipe_version(recipe_id),
        "train_filename": "train.jsonl",
        "validation_filename": "validation.jsonl",
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "train_sha256": train_hash,
        "validation_sha256": val_hash,
        "model_visible_top_level_fields": list(
            RECIPE_TOP_LEVEL_FIELDS[recipe_id]
        ),
        "schema_filename": "schema.json",
        "schema_sha256": schema_hash,
        "example_filename": "EXAMPLE.jsonl",
        "example_sha256": sha256_file(trainer_dir / "EXAMPLE.jsonl"),
    }
    trainer_manifest_hash = _write_json(
        trainer_dir / "manifest.json", trainer_manifest
    )
    dataset_manifest = {
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "train_sha256": train_hash,
        "validation_sha256": val_hash,
        "dataset_version": dataset_version,
        "record_format": record_format,
        "recipe_id": recipe_id,
        "model_visible_top_level_fields": list(
            RECIPE_TOP_LEVEL_FIELDS[recipe_id]
        ),
        "schema_sha256": schema_hash,
        "record_map_ref": "provenance/record-map.jsonl",
        "record_map_sha256": record_map_hash,
        "trainer_upload_manifest_sha256": trainer_manifest_hash,
    }
    _write_json(dataset_dir / "manifest.json", dataset_manifest)

    # Eval pack (EvalFoundry-compatible minimal surface)
    pack_dir = eval_dir / "pack"
    _write_jsonl(pack_dir / "cases.jsonl", [{k: v for k, v in c.items() if k != "label"} for c in eval_cases])
    _write_jsonl(pack_dir / "answers.jsonl", [{"id": c["id"], "label": c["label"]} for c in eval_cases])
    grader = {
        "grader_id": "exact_label_match_v1",
        "type": "deterministic",
        "description": "Compares model or fixture output label to sealed answer label by case id.",
    }
    _write_json(eval_dir / "graders" / "grader.json", grader)
    baselines = {"fixture": {"notes": "fixture_only product — not a commercial baseline"}}
    _write_json(eval_dir / "baselines" / "baselines.json", baselines)
    eval_manifest = {
        "eval_version": eval_version,
        "case_count": len(eval_cases),
        "grader_id": grader["grader_id"],
        "dataset_version": dataset_version,
    }
    _write_json(eval_dir / "manifest.json", eval_manifest)

    # Curation ledger
    _write_json(curation_dir / "rubric.json", rubric)
    decisions_hash = _write_jsonl(curation_dir / "decisions.jsonl", decisions)
    review_manifest = {
        "considered": len(ingots),
        "included": len(included),
        "excluded": len(excluded),
        "held": len(held),
        "reconciliation": "considered = included + excluded + held",
    }
    _write_json(curation_dir / "review-manifest.json", review_manifest)

    # Provenance
    source_ingots = [
        {
            "ingot_id": i["ingot_id"],
            "ingot_hash": sha256_json(i),
            "source_class": i.get("source_class"),
            "source_bundle_hash": i.get("source_bundle_hash"),
            "mine_run_id": i.get("mine_run_id"),
            "domain_claim_id": i.get("domain_claim_id"),
            "domain_claim": i.get("domain_claim"),
            "interactive_evidence": (
                (i.get("deterministic_results") or {}).get("interactive_evidence")
                if isinstance(i.get("deterministic_results"), dict)
                else None
            ),
            "scenario_trace_evidence": (
                (i.get("deterministic_results") or {}).get(
                    "scenario_trace_evidence"
                )
                if isinstance(i.get("deterministic_results"), dict)
                else None
            ),
            "processor_routing": (
                (i.get("deterministic_results") or {}).get("processor_routing")
                if isinstance(i.get("deterministic_results"), dict)
                else None
            ),
        }
        for i in ingots
    ]
    _write_jsonl(provenance_dir / "source-ingots.jsonl", source_ingots)
    lineage = [
        {
            "schema_id": "goldtrace.lineage.event.v1",
            "event_id": f"lin-{i['ingot_id']}",
            "stage": "mint",
            "action": "include" if any(d["ingot_id"] == i["ingot_id"] and d["decision"] == "INCLUDE" for d in decisions) else "consider",
            "subject_hash": sha256_json(i),
            "parent_hashes": [i.get("source_bundle_hash") or ""],
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metadata": {
                "ingot_id": i["ingot_id"],
                "domain_claim_id": i.get("domain_claim_id"),
            },
        }
        for i in ingots
    ]
    _write_jsonl(provenance_dir / "lineage.jsonl", lineage)
    declared_license = (product_spec.get("rights_requirements") or {}).get("license")
    rights = {
        "license": declared_license or "UNSPECIFIED",
        "fixture_only": bool(product_spec.get("fixture_only", False)),
        "rights_status": (
            "fixture"
            if product_spec.get("fixture_only")
            else ("declared" if declared_license else "unknown")
        ),
        "sources": source_ingots,
    }
    rights_hash = _write_json(provenance_dir / "rights.json", rights)

    # Reports
    (reports_dir / "DATASET_CARD.md").write_text(
        f"# Dataset Card\n\nProduct: {product_spec.get('product_id')}@{dataset_version}\n\n"
        f"Fixture only: {product_spec.get('fixture_only')}\n\n"
        f"Train rows: {len(train_rows)}\nValidation rows: {len(val_rows)}\n",
        encoding="utf-8",
    )
    (reports_dir / "EVAL_CARD.md").write_text(
        f"# Eval Card\n\nEval version: {eval_version}\nCases: {len(eval_cases)}\n"
        f"Grader: {grader['grader_id']}\n",
        encoding="utf-8",
    )
    (reports_dir / "QUALITY_REPORT.md").write_text(
        f"# Quality Report\n\nIncluded: {len(included)}\nExcluded: {len(excluded)}\nHeld: {len(held)}\n"
        f"This product is fixture_only={product_spec.get('fixture_only')} and is not commercially approved.\n",
        encoding="utf-8",
    )

    product_spec_hash = sha256_json(product_spec)
    _write_json(out_dir / "product-spec.json", product_spec)

    # SHA256SUMS over product files
    sums_lines = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(out_dir).as_posix()
        if rel == "SHA256SUMS":
            continue
        sums_lines.append(f"{sha256_file(path)}  {rel}")
    (out_dir / "SHA256SUMS").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_id": "goldtrace.mint.product_lot.v1",
        "product_id": product_spec["product_id"],
        "product_version": product_spec["product_version"],
        "dataset": dataset_manifest,
        "eval": eval_manifest,
        "product_spec_hash": product_spec_hash,
        "curation_ledger_hash": decisions_hash,
        "source_ingot_ledger_hash": sha256_file(provenance_dir / "source-ingots.jsonl"),
        "lineage_manifest_hash": sha256_file(provenance_dir / "lineage.jsonl"),
        "rights_manifest_hash": rights_hash,
        # Written as a placeholder, then replaced by the canonical lot hash.
        # The hash algorithm covers every manifest field except this one.
        "product_hash": "",
        "fixture_only": bool(product_spec.get("fixture_only", False)),
        "dataset_version": dataset_version,
        "eval_version": eval_version,
    }
    _write_json(out_dir / "product-manifest.json", manifest)
    product_hash = sha256_product_lot(out_dir)
    manifest["product_hash"] = product_hash
    _write_json(out_dir / "product-manifest.json", manifest)
    if sha256_product_lot(out_dir) != product_hash:
        raise RuntimeError("canonical product lot hash changed after manifest binding")

    return {
        "product_lot": str(out_dir),
        "product_hash": product_hash,
        "manifest": manifest,
        "counts": review_manifest,
    }


def verify_complete(product_lot: Path) -> list[str]:
    root = Path(product_lot).resolve()
    errors: list[str] = []
    required = [
        "dataset/train.jsonl",
        "dataset/validation.jsonl",
        "dataset/schema.json",
        "dataset/manifest.json",
        "eval/pack/cases.jsonl",
        "eval/pack/answers.jsonl",
        "eval/graders/grader.json",
        "eval/manifest.json",
        "curation/decisions.jsonl",
        "curation/rubric.json",
        "provenance/source-ingots.jsonl",
        "provenance/lineage.jsonl",
        "provenance/record-map.jsonl",
        "provenance/rights.json",
        "trainer-upload/train.jsonl",
        "trainer-upload/validation.jsonl",
        "trainer-upload/schema.json",
        "trainer-upload/manifest.json",
        "trainer-upload/README.md",
        "trainer-upload/EXAMPLE.jsonl",
        "product-spec.json",
        "product-manifest.json",
        "SHA256SUMS",
    ]
    for rel in required:
        if not (root / rel).exists():
            errors.append(f"missing {rel}")
    if (root / "eval").exists() is False:
        errors.append("product lot without an eval pack")

    def load_jsonl(relative: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in (root / relative).read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{relative} contains a non-object row")
                rows.append(value)
        return rows

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    record_maps: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    source_ingots: list[dict[str, Any]] = []
    for relative, target in (
        ("dataset/train.jsonl", train_rows),
        ("dataset/validation.jsonl", validation_rows),
        ("provenance/record-map.jsonl", record_maps),
        ("curation/decisions.jsonl", decisions),
        ("provenance/source-ingots.jsonl", source_ingots),
    ):
        if not (root / relative).is_file():
            continue
        try:
            target.extend(load_jsonl(relative))
        except Exception as exc:
            errors.append(f"invalid {relative}: {exc}")
    if not decisions:
        errors.append("product lot without curation decisions")

    for relative, rows in (
        ("dataset/train.jsonl", train_rows),
        ("dataset/validation.jsonl", validation_rows),
    ):
        for index, row in enumerate(rows):
            if "_content_warning" in row:
                errors.append(f"provenance-only row forbidden: {relative}[{index}]")

    dataset_manifest: dict[str, Any] = {}
    dataset_schema: dict[str, Any] = {}
    trainer_manifest: dict[str, Any] = {}
    for relative, label in (
        ("dataset/manifest.json", "dataset manifest"),
        ("dataset/schema.json", "dataset schema"),
        ("trainer-upload/manifest.json", "trainer-upload manifest"),
    ):
        if not (root / relative).is_file():
            continue
        try:
            value = json.loads((root / relative).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("expected object")
            if relative == "dataset/manifest.json":
                dataset_manifest = value
            elif relative == "dataset/schema.json":
                dataset_schema = value
            else:
                trainer_manifest = value
        except Exception as exc:
            errors.append(f"invalid {label}: {exc}")

    if dataset_manifest:
        if dataset_manifest.get("train_count") != len(train_rows):
            errors.append("dataset train count differs from train.jsonl")
        if dataset_manifest.get("validation_count") != len(validation_rows):
            errors.append("dataset validation count differs from validation.jsonl")
        train_path = root / "dataset/train.jsonl"
        if train_path.is_file() and dataset_manifest.get(
            "train_sha256"
        ) != sha256_file(train_path):
            errors.append("dataset train hash differs from train.jsonl")
        validation_path = root / "dataset/validation.jsonl"
        if validation_path.is_file() and dataset_manifest.get(
            "validation_sha256"
        ) != sha256_file(validation_path):
            errors.append("dataset validation hash differs from validation.jsonl")

    if dataset_schema:
        validator = Draft202012Validator(dataset_schema)
        for split, rows in (("train", train_rows), ("validation", validation_rows)):
            for index, row in enumerate(rows):
                schema_errors = list(validator.iter_errors(row))
                if schema_errors:
                    errors.append(
                        f"{split} row {index} violates dataset schema: "
                        f"{schema_errors[0].message}"
                    )
        declared_fields = dataset_manifest.get("model_visible_top_level_fields")
        if isinstance(declared_fields, list):
            expected_fields = set(declared_fields)
            for split, rows in (("train", train_rows), ("validation", validation_rows)):
                for index, row in enumerate(rows):
                    if set(row) != expected_fields:
                        errors.append(
                            f"{split} row {index} contains fields outside its model surface"
                        )
        if dataset_manifest.get("schema_sha256") != sha256_file(
            root / "dataset/schema.json"
        ):
            errors.append("dataset schema hash binding differs")

    map_validator = Draft202012Validator(_contract("mint-record-map.v1.schema.json"))
    expected_map_count = len(train_rows) + len(validation_rows)
    if len(record_maps) != expected_map_count:
        errors.append(
            f"record map count differs: expected={expected_map_count} actual={len(record_maps)}"
        )
    seen_positions: set[tuple[str, int]] = set()
    seen_record_ids: set[str] = set()
    source_by_id = {
        source.get("ingot_id"): source
        for source in source_ingots
        if isinstance(source.get("ingot_id"), str)
    }
    decision_by_id = {
        decision.get("decision_id"): decision
        for decision in decisions
        if isinstance(decision.get("decision_id"), str)
    }
    if len(source_by_id) != len(source_ingots):
        errors.append("source ingot ledger has missing or duplicate ingot ids")
    if len(decision_by_id) != len(decisions):
        errors.append("curation ledger has missing or duplicate decision ids")

    def rendered_tool_ids(row: dict[str, Any]) -> tuple[list[str], list[str]]:
        requests: list[str] = []
        results: list[str] = []
        messages = row.get("messages")
        if not isinstance(messages, list):
            return requests, results
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                calls = message.get("tool_calls")
                if isinstance(calls, list):
                    requests.extend(
                        call["id"]
                        for call in calls
                        if isinstance(call, dict) and isinstance(call.get("id"), str)
                    )
            elif message.get("role") == "tool" and isinstance(
                message.get("tool_call_id"), str
            ):
                results.append(message["tool_call_id"])
        return requests, results

    for index, mapping in enumerate(record_maps):
        map_errors = list(map_validator.iter_errors(mapping))
        if map_errors:
            errors.append(f"record map {index} violates contract: {map_errors[0].message}")
            continue
        split = mapping["split"]
        row_index = mapping["row_index"]
        position = (split, row_index)
        if position in seen_positions:
            errors.append(f"duplicate provenance record-map position: {position}")
            continue
        seen_positions.add(position)
        record_id = mapping["record_id"]
        if record_id in seen_record_ids:
            errors.append(f"duplicate provenance record id: {record_id}")
        seen_record_ids.add(record_id)
        rows = train_rows if split == "train" else validation_rows
        if row_index >= len(rows):
            errors.append(f"record map {index} points outside {split}.jsonl")
            continue
        if mapping["rendered_row_sha256"] != sha256_bytes(
            _jsonl_line(rows[row_index])
        ):
            errors.append(f"record map {index} rendered-row hash differs")
        expected_path = (
            "dataset/train.jsonl"
            if split == "train"
            else "dataset/validation.jsonl"
        )
        if mapping["dataset_path"] != expected_path:
            errors.append(f"record map {index} dataset path differs")

        source = source_by_id.get(mapping["source_ingot_id"])
        if not isinstance(source, dict):
            errors.append(f"record map {index} references an unknown source ingot")
        else:
            for map_field, source_field in (
                ("source_ingot_hash", "ingot_hash"),
                ("source_bundle_hash", "source_bundle_hash"),
                ("mine_run_id", "mine_run_id"),
            ):
                if mapping[map_field] != source.get(source_field):
                    errors.append(
                        f"record map {index} {map_field} differs from source ledger"
                    )
            interactive_hash = mapping["source_interactive_evidence_sha256"]
            source_index_hash = mapping["source_index_sha256"]
            if interactive_hash is not None or source_index_hash is not None:
                descriptor = source.get("interactive_evidence")
                if not isinstance(descriptor, dict):
                    errors.append(
                        f"record map {index} lacks its interactive source descriptor"
                    )
                elif (
                    descriptor.get("sha256") != interactive_hash
                    or descriptor.get("source_index_sha256") != source_index_hash
                ):
                    errors.append(
                        f"record map {index} interactive source hashes differ"
                    )
            metadata = mapping.get("extractor_metadata")
            if isinstance(metadata, dict) and metadata.get(
                "source_scenario_trace_evidence_sha256"
            ) is not None:
                descriptor = source.get("scenario_trace_evidence")
                routing = source.get("processor_routing")
                expected = {
                    "source_scenario_trace_evidence_sha256": (
                        descriptor.get("sha256") if isinstance(descriptor, dict) else None
                    ),
                    "source_trace_artifact_id": (
                        descriptor.get("trace_artifact_id")
                        if isinstance(descriptor, dict)
                        else None
                    ),
                    "source_trace_file_sha256": (
                        descriptor.get("source_trace_file_sha256")
                        if isinstance(descriptor, dict)
                        else None
                    ),
                    "source_trace_content_sha256": (
                        descriptor.get("source_trace_content_sha256")
                        if isinstance(descriptor, dict)
                        else None
                    ),
                    "scenario_id": (
                        descriptor.get("scenario_id")
                        if isinstance(descriptor, dict)
                        else None
                    ),
                    "source_family": (
                        routing.get("source_family")
                        if isinstance(routing, dict)
                        else None
                    ),
                }
                if not isinstance(descriptor, dict) or not isinstance(routing, dict):
                    errors.append(
                        f"record map {index} lacks its scenario-trace source descriptors"
                    )
                elif any(metadata.get(key) != value for key, value in expected.items()):
                    errors.append(
                        f"record map {index} scenario-trace source lineage differs"
                    )

        decision = decision_by_id.get(mapping["curation_decision_id"])
        if (
            not isinstance(decision, dict)
            or decision.get("ingot_id") != mapping["source_ingot_id"]
            or decision.get("ingot_hash") != mapping["source_ingot_hash"]
            or decision.get("split_group_id") != mapping["split_group_id"]
        ):
            errors.append(f"record map {index} differs from its curation decision")
        if dataset_manifest and mapping["recipe_id"] != dataset_manifest.get(
            "recipe_id"
        ):
            errors.append(f"record map {index} recipe differs from dataset")

        normalized_calls = mapping["normalized_tool_calls"]
        if mapping["source_interactive_evidence_sha256"] is not None:
            normalized_ids = [call["normalized_call_id"] for call in normalized_calls]
            request_ids, result_ids = rendered_tool_ids(rows[row_index])
            if not normalized_ids or request_ids != normalized_ids or result_ids != normalized_ids:
                errors.append(
                    f"record map {index} normalized tool ids differ from rendered row"
                )
            if mapping["request_event_ids"] != [
                call["request_event_id"] for call in normalized_calls
            ]:
                errors.append(f"record map {index} tool request references differ")
            if mapping["result_event_ids"] != [
                call["result_event_id"] for call in normalized_calls
            ]:
                errors.append(f"record map {index} tool result references differ")
            if mapping["effect_event_ids"] != [
                event_id
                for call in normalized_calls
                for event_id in call["effect_event_ids"]
            ]:
                errors.append(f"record map {index} tool effect references differ")

    expected_positions = {
        *(("train", index) for index in range(len(train_rows))),
        *(("validation", index) for index in range(len(validation_rows))),
    }
    if seen_positions != expected_positions:
        errors.append("record map does not cover the exact dataset row positions")
    record_map_path = root / "provenance/record-map.jsonl"
    if (
        dataset_manifest
        and record_map_path.is_file()
        and dataset_manifest.get("record_map_sha256") != sha256_file(record_map_path)
    ):
        errors.append("dataset record-map hash binding differs")

    for name in ("train.jsonl", "validation.jsonl", "schema.json"):
        dataset_file = root / "dataset" / name
        trainer_file = root / "trainer-upload" / name
        if dataset_file.is_file() and trainer_file.is_file():
            if dataset_file.read_bytes() != trainer_file.read_bytes():
                errors.append(f"trainer-upload/{name} differs from dataset/{name}")

    if trainer_manifest:
        expected_trainer = {
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
        }
        for field, relative in (
            ("train_sha256", "trainer-upload/train.jsonl"),
            ("validation_sha256", "trainer-upload/validation.jsonl"),
            ("schema_sha256", "trainer-upload/schema.json"),
            ("example_sha256", "trainer-upload/EXAMPLE.jsonl"),
        ):
            path = root / relative
            if path.is_file():
                expected_trainer[field] = sha256_file(path)
        for field, expected in expected_trainer.items():
            if trainer_manifest.get(field) != expected:
                errors.append(f"trainer-upload manifest {field} differs")
        if dataset_manifest.get("trainer_upload_manifest_sha256") != sha256_file(
            root / "trainer-upload/manifest.json"
        ):
            errors.append("dataset trainer-upload manifest hash binding differs")
        if trainer_manifest.get("model_visible_top_level_fields") != dataset_manifest.get(
            "model_visible_top_level_fields"
        ):
            errors.append("trainer and dataset model-visible fields differ")

    sums_path = root / "SHA256SUMS"
    if sums_path.is_file():
        declared: dict[str, str] = {}
        try:
            for line in sums_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                digest, relative = line.split(None, 1)
                relative = relative.strip()
                if relative in declared:
                    raise ValueError(f"duplicate path {relative}")
                declared[relative] = digest
            expected_paths = {
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and path.relative_to(root).as_posix()
                not in {"SHA256SUMS", "product-manifest.json"}
            }
            if set(declared) != expected_paths:
                errors.append("SHA256SUMS does not cover the exact product payload set")
            for relative, digest in declared.items():
                path = root / relative
                if not path.is_file() or sha256_file(path) != digest:
                    errors.append(f"SHA256SUMS digest differs: {relative}")
                    break
        except Exception as exc:
            errors.append(f"invalid SHA256SUMS: {exc}")

    manifest_path = root / "product-manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("product_hash") != sha256_product_lot(root):
                errors.append("product manifest hash differs from exact product lot")
        except Exception as exc:
            errors.append(f"invalid product hash binding: {exc}")
    return errors
