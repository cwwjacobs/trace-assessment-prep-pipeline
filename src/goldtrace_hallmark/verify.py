from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from .hashing import sha256_file, sha256_json, sha256_product_lot, sha256_tree


ReviewerFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _contract(name: str) -> dict[str, Any]:
    """Load a contract from package data when installed, or the checkout in dev.

    Walking parent directories for a ``contracts/`` sibling only works inside a
    source tree. Hallmark decides whether a product passes, so it has to resolve
    its contracts from an installed release too, and the shared resolver is the
    only lookup that reaches package data.
    """
    from goldtrace_refinery.paths import find_contract

    return json.loads(find_contract(name).read_text(encoding="utf-8"))


def _schema_errors(instance: Any, contract_name: str) -> list[str]:
    validator = Draft202012Validator(_contract(contract_name))
    return [error.message for error in sorted(validator.iter_errors(instance), key=lambda item: list(item.path))]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def fixture_semantic_reviewer(ingot_meta: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
    """Fixture reviewer — may only support fixture_only products."""
    return {
        "disposition": "ACCEPT",
        "reviewer_type": "fixture",
        "notes": "fixture_only semantic re-review",
        "rubric_id": rubric.get("rubric_id"),
    }


def production_semantic_reviewer(ingot_meta: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
    """Represent the missing production reviewer honestly as a review HOLD."""
    return {
        "disposition": "HOLD",
        "reviewer_type": "unconfigured",
        "blocked_reason": "production_semantic_reviewer_not_configured",
        "notes": "No production semantic reviewer has been delegated by the operator.",
        "rubric_id": rubric.get("rubric_id"),
    }


def verify_product_lot(
    product_lot: Path,
    *,
    mode: str = "fixture",
    semantic_reviewer: ReviewerFn | None = None,
) -> dict[str, Any]:
    """Independent Hallmark verification. Does not mutate the product lot."""
    root = Path(product_lot).resolve()
    checks: list[dict[str, Any]] = []
    discrepancies: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            discrepancies.append({"check": name, "detail": detail})

    try:
        initial_tree_hash = sha256_tree(root)
        initial_hash_error = ""
    except Exception as exc:
        initial_tree_hash = ""
        initial_hash_error = str(exc)

    # Structural
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
        add_check(f"exists:{rel}", (root / rel).exists(), rel)

    manifest = json.loads((root / "product-manifest.json").read_text(encoding="utf-8")) if (root / "product-manifest.json").exists() else {}
    product_spec = json.loads((root / "product-spec.json").read_text(encoding="utf-8")) if (root / "product-spec.json").exists() else {}
    rubric = json.loads((root / "curation" / "rubric.json").read_text(encoding="utf-8")) if (root / "curation" / "rubric.json").exists() else {}
    fixture_only = bool(manifest.get("fixture_only") or product_spec.get("fixture_only"))

    try:
        manifest_errors = _schema_errors(manifest, "product-lot.v1.schema.json")
        add_check("product_manifest_schema", not manifest_errors, "; ".join(manifest_errors))
    except Exception as exc:
        add_check("product_manifest_schema", False, str(exc))

    # JSONL parse
    train_rows, validation_rows, decisions, sources, record_maps = [], [], [], [], []
    try:
        train_rows = _load_jsonl(root / "dataset" / "train.jsonl")
        add_check("train_jsonl_parse", True, f"count={len(train_rows)}")
    except Exception as exc:
        add_check("train_jsonl_parse", False, str(exc))
    try:
        validation_rows = _load_jsonl(root / "dataset" / "validation.jsonl")
        add_check("validation_jsonl_parse", True, f"count={len(validation_rows)}")
    except Exception as exc:
        add_check("validation_jsonl_parse", False, str(exc))
    try:
        decisions = _load_jsonl(root / "curation" / "decisions.jsonl")
        add_check("decisions_jsonl_parse", True, f"count={len(decisions)}")
    except Exception as exc:
        add_check("decisions_jsonl_parse", False, str(exc))
    try:
        sources = _load_jsonl(root / "provenance" / "source-ingots.jsonl")
        add_check("source_ingots_parse", True, f"count={len(sources)}")
    except Exception as exc:
        add_check("source_ingots_parse", False, str(exc))
    try:
        record_maps = _load_jsonl(root / "provenance" / "record-map.jsonl")
        add_check("record_map_parse", True, f"count={len(record_maps)}")
    except Exception as exc:
        add_check("record_map_parse", False, str(exc))

    # Counts
    ds_manifest = json.loads((root / "dataset" / "manifest.json").read_text(encoding="utf-8")) if (root / "dataset" / "manifest.json").exists() else {}
    add_check(
        "train_count_match",
        ds_manifest.get("train_count") == len(train_rows),
        f"declared={ds_manifest.get('train_count')} actual={len(train_rows)}",
    )
    add_check(
        "validation_count_match",
        ds_manifest.get("validation_count") == len(validation_rows),
        f"declared={ds_manifest.get('validation_count')} actual={len(validation_rows)}",
    )

    # Substance. Every check above this point verifies integrity, and integrity
    # is satisfied trivially by nothing at all: an empty payload matches the
    # hash of an empty payload, parses without a single bad line, declares a
    # count of zero that its manifest agrees with, and contains no secrets. A
    # lot shipped with e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    # for both splits passed every one of them and was admitted to the vault.
    #
    # These checks ask the separate question: is there anything here? They are
    # named apart from the bindings so a report can say integrity OK, substance
    # FAILED, which is the distinction a buyer is paying for.
    substance = rubric.get("substance")
    if not isinstance(substance, dict):
        substance = {}

    def _positive_int(key: str, floor: int) -> int:
        """Read a rubric threshold that may raise the floor but never lower it."""
        try:
            declared = int(substance.get(key, floor))
        except (TypeError, ValueError):
            return floor
        return max(floor, declared)

    # A rubric may demand more than one record. It may not demand fewer: no
    # rubric is permitted to authorise an empty product.
    min_train_records = _positive_int("min_train_records", 1)
    min_turns_per_record = _positive_int("min_turns_per_record", 0)
    required_roles = [
        role for role in (substance.get("required_roles") or []) if isinstance(role, str)
    ]

    add_check(
        "substance.train_not_empty",
        bool(train_rows),
        f"train rows={len(train_rows)}",
    )
    add_check(
        "substance.train_min_records",
        len(train_rows) >= min_train_records,
        f"required>={min_train_records} actual={len(train_rows)}",
    )

    # An empty validation split is legitimate only when the split policy never
    # asked for one.
    try:
        validation_ratio = float(
            (product_spec.get("split_policy") or {}).get("validation_ratio") or 0
        )
    except (TypeError, ValueError):
        validation_ratio = 0.0
    add_check(
        "substance.validation_present_when_declared",
        bool(validation_rows) or validation_ratio <= 0,
        f"validation_ratio={validation_ratio} rows={len(validation_rows)}",
    )

    def _messages(row: object) -> list[dict[str, Any]] | None:
        """Return the message list for chat-shaped rows, or None for other recipes."""
        if not isinstance(row, dict):
            return None
        messages = row.get("messages")
        if not isinstance(messages, list):
            return None
        return [message for message in messages if isinstance(message, dict)]

    chat_rows = [
        (split, index, messages)
        for split, rows in (("train", train_rows), ("validation", validation_rows))
        for index, row in enumerate(rows)
        if (messages := _messages(row)) is not None
    ]

    if min_turns_per_record > 0:
        thin = [
            f"{split}[{index}]={len(messages)}"
            for split, index, messages in chat_rows
            if len(messages) < min_turns_per_record
        ]
        add_check(
            "substance.min_turns_per_record",
            not thin,
            f"required>={min_turns_per_record} under={thin}"
            if thin
            else f"required>={min_turns_per_record} checked={len(chat_rows)}",
        )

    if required_roles:
        # A rubric that names required roles against a recipe carrying no
        # messages is asking for something the format cannot express.
        role_errors: list[str] = []
        if not chat_rows:
            role_errors.append("no message-shaped records to carry the required roles")
        for split, index, messages in chat_rows:
            present = {message.get("role") for message in messages}
            missing = [role for role in required_roles if role not in present]
            if missing:
                role_errors.append(f"{split}[{index}] missing={missing}")
        add_check(
            "substance.required_roles_present",
            not role_errors,
            f"required={required_roles} " + ("; ".join(role_errors[:5]) or "satisfied"),
        )

    # Model-facing rows are deliberately provenance-free. The independently
    # hashed record map carries source identity and binds each exact JSONL row.
    record_map_errors: list[str] = []
    record_positions: set[tuple[str, int]] = set()
    record_ids: set[str] = set()
    source_by_id = {
        source.get("ingot_id"): source
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("ingot_id"), str)
    }
    decision_by_id = {
        decision.get("decision_id"): decision
        for decision in decisions
        if isinstance(decision, dict)
        and isinstance(decision.get("decision_id"), str)
    }
    if len(source_by_id) != len(sources):
        record_map_errors.append("source ledger has missing or duplicate ingot ids")
    if len(decision_by_id) != len(decisions):
        record_map_errors.append("curation ledger has missing or duplicate decision ids")

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
        schema_errors = _schema_errors(mapping, "mint-record-map.v1.schema.json")
        if schema_errors:
            record_map_errors.append(
                f"record_map[{index}] schema: {'; '.join(schema_errors)}"
            )
            continue
        split = mapping.get("split")
        row_index = mapping.get("row_index")
        position = (split, row_index)
        if position in record_positions:
            record_map_errors.append(f"duplicate record-map position: {position}")
            continue
        record_positions.add(position)
        record_id = mapping["record_id"]
        if record_id in record_ids:
            record_map_errors.append(f"duplicate record id: {record_id}")
        record_ids.add(record_id)
        rows = train_rows if split == "train" else validation_rows
        if not isinstance(row_index, int) or row_index < 0 or row_index >= len(rows):
            record_map_errors.append(f"record_map[{index}] points outside {split}")
            continue
        rendered = (
            json.dumps(rows[row_index], sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        actual_row_hash = hashlib.sha256(rendered).hexdigest()
        if mapping.get("rendered_row_sha256") != actual_row_hash:
            record_map_errors.append(f"record_map[{index}] row hash mismatch")
        expected_path = (
            "dataset/train.jsonl"
            if split == "train"
            else "dataset/validation.jsonl"
        )
        if mapping.get("dataset_path") != expected_path:
            record_map_errors.append(f"record_map[{index}] dataset path mismatch")

        source = source_by_id.get(mapping["source_ingot_id"])
        if not isinstance(source, dict):
            record_map_errors.append(f"record_map[{index}] unknown source ingot")
        else:
            for map_field, source_field in (
                ("source_ingot_hash", "ingot_hash"),
                ("source_bundle_hash", "source_bundle_hash"),
                ("mine_run_id", "mine_run_id"),
            ):
                if mapping[map_field] != source.get(source_field):
                    record_map_errors.append(
                        f"record_map[{index}] {map_field} source mismatch"
                    )
            interactive_hash = mapping["source_interactive_evidence_sha256"]
            source_index_hash = mapping["source_index_sha256"]
            if interactive_hash is not None or source_index_hash is not None:
                descriptor = source.get("interactive_evidence")
                if not isinstance(descriptor, dict) or (
                    descriptor.get("sha256") != interactive_hash
                    or descriptor.get("source_index_sha256") != source_index_hash
                ):
                    record_map_errors.append(
                        f"record_map[{index}] interactive source hash mismatch"
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
                    record_map_errors.append(
                        f"record_map[{index}] lacks scenario-trace source descriptors"
                    )
                elif any(metadata.get(key) != value for key, value in expected.items()):
                    record_map_errors.append(
                        f"record_map[{index}] scenario-trace lineage mismatch"
                    )

        decision = decision_by_id.get(mapping["curation_decision_id"])
        if (
            not isinstance(decision, dict)
            or decision.get("ingot_id") != mapping["source_ingot_id"]
            or decision.get("ingot_hash") != mapping["source_ingot_hash"]
            or decision.get("split_group_id") != mapping["split_group_id"]
        ):
            record_map_errors.append(f"record_map[{index}] curation mismatch")
        if mapping["recipe_id"] != ds_manifest.get("recipe_id"):
            record_map_errors.append(f"record_map[{index}] recipe mismatch")

        normalized_calls = mapping["normalized_tool_calls"]
        if mapping["source_interactive_evidence_sha256"] is not None:
            normalized_ids = [call["normalized_call_id"] for call in normalized_calls]
            request_ids, result_ids = rendered_tool_ids(rows[row_index])
            if not normalized_ids or request_ids != normalized_ids or result_ids != normalized_ids:
                record_map_errors.append(
                    f"record_map[{index}] rendered tool correlation mismatch"
                )
            if mapping["request_event_ids"] != [
                call["request_event_id"] for call in normalized_calls
            ]:
                record_map_errors.append(
                    f"record_map[{index}] request-event correlation mismatch"
                )
            if mapping["result_event_ids"] != [
                call["result_event_id"] for call in normalized_calls
            ]:
                record_map_errors.append(
                    f"record_map[{index}] result-event correlation mismatch"
                )
            if mapping["effect_event_ids"] != [
                event_id
                for call in normalized_calls
                for event_id in call["effect_event_ids"]
            ]:
                record_map_errors.append(
                    f"record_map[{index}] effect-event correlation mismatch"
                )
    expected_positions = {
        *(("train", index) for index in range(len(train_rows))),
        *(("validation", index) for index in range(len(validation_rows))),
    }
    if record_positions != expected_positions:
        record_map_errors.append(
            f"record-map positions differ: expected={sorted(expected_positions)} "
            f"actual={sorted(record_positions)}"
        )
    add_check(
        "record_map_integrity",
        not record_map_errors,
        "; ".join(record_map_errors),
    )

    try:
        dataset_schema = json.loads(
            (root / "dataset" / "schema.json").read_text(encoding="utf-8")
        )
        row_validator = Draft202012Validator(dataset_schema)
        row_schema_errors: list[str] = []
        for split, rows in (("train", train_rows), ("validation", validation_rows)):
            for index, row in enumerate(rows):
                errors = list(row_validator.iter_errors(row))
                if errors:
                    row_schema_errors.append(
                        f"{split}[{index}]: {errors[0].message}"
                    )
        add_check(
            "model_visible_schema",
            not row_schema_errors,
            "; ".join(row_schema_errors),
        )
    except Exception as exc:
        add_check("model_visible_schema", False, str(exc))

    # Curation: included records must have INCLUDE decisions; excluded not in train
    include_ids = {d["ingot_id"] for d in decisions if d.get("decision") == "INCLUDE"}
    exclude_ids = {d["ingot_id"] for d in decisions if d.get("decision") == "EXCLUDE"}
    train_ingots = {
        mapping.get("source_ingot_id")
        for mapping in record_maps
        if mapping.get("split") == "train"
        and isinstance(mapping.get("source_ingot_id"), str)
    }
    add_check("included_have_decisions", train_ingots <= include_ids, f"extra={sorted(train_ingots - include_ids)}")
    add_check("excluded_not_in_train", train_ingots.isdisjoint(exclude_ids), f"leaked={sorted(train_ingots & exclude_ids)}")

    # Source resolution
    source_ids = {s.get("ingot_id") for s in sources}
    unresolved = train_ingots - source_ids
    add_check("source_ingot_resolvable", not unresolved, f"missing={sorted(unresolved)}")

    # Leakage groups
    eval_cases: list[dict[str, Any]] = []
    eval_answers: list[dict[str, Any]] = []
    try:
        eval_cases = _load_jsonl(root / "eval" / "pack" / "cases.jsonl")
        add_check("eval_cases_loadable", True, f"count={len(eval_cases)}")
    except Exception as exc:
        add_check("eval_cases_loadable", False, str(exc))
    try:
        eval_answers = _load_jsonl(root / "eval" / "pack" / "answers.jsonl")
        add_check("eval_answers_loadable", True, f"count={len(eval_answers)}")
    except Exception as exc:
        add_check("eval_answers_loadable", False, str(exc))
    train_groups = {
        mapping.get("split_group_id")
        for mapping in record_maps
        if mapping.get("split") == "train"
        and isinstance(mapping.get("split_group_id"), str)
    }
    eval_groups = {c.get("split_group_id") for c in eval_cases if c.get("split_group_id")}
    leak = train_groups & eval_groups
    add_check("no_split_leakage", not leak, f"groups={sorted(g for g in leak if g)}")

    # Version correspondence
    eval_manifest = json.loads((root / "eval" / "manifest.json").read_text(encoding="utf-8")) if (root / "eval" / "manifest.json").exists() else {}
    add_check(
        "dataset_eval_version_correspondence",
        eval_manifest.get("dataset_version") == manifest.get("dataset_version")
        and eval_manifest.get("eval_version") == manifest.get("eval_version"),
        f"dataset={manifest.get('dataset_version')} eval={eval_manifest.get('eval_version')}",
    )

    # Manifest reference integrity. Exact lot hashing binds bytes, but these
    # checks independently prove that the manifest's internal references tell
    # the truth about those bytes.
    def add_hash_binding(name: str, declared: object, path: Path) -> None:
        try:
            actual = sha256_file(path)
            ok = isinstance(declared, str) and declared == actual
            add_check(name, ok, f"declared={declared} actual={actual}")
        except Exception as exc:
            add_check(name, False, str(exc))

    add_check(
        "dataset_manifest_binding",
        manifest.get("dataset") == ds_manifest,
        "product manifest dataset object must equal dataset/manifest.json",
    )
    add_check(
        "eval_manifest_binding",
        manifest.get("eval") == eval_manifest,
        "product manifest eval object must equal eval/manifest.json",
    )
    add_check(
        "product_spec_hash_binding",
        manifest.get("product_spec_hash") == sha256_json(product_spec),
        f"declared={manifest.get('product_spec_hash')} actual={sha256_json(product_spec)}",
    )
    add_hash_binding(
        "curation_ledger_hash_binding",
        manifest.get("curation_ledger_hash"),
        root / "curation" / "decisions.jsonl",
    )
    add_hash_binding(
        "source_ingot_ledger_hash_binding",
        manifest.get("source_ingot_ledger_hash"),
        root / "provenance" / "source-ingots.jsonl",
    )
    add_hash_binding(
        "lineage_manifest_hash_binding",
        manifest.get("lineage_manifest_hash"),
        root / "provenance" / "lineage.jsonl",
    )
    add_hash_binding(
        "rights_manifest_hash_binding",
        manifest.get("rights_manifest_hash"),
        root / "provenance" / "rights.json",
    )
    add_hash_binding(
        "train_payload_hash_binding",
        ds_manifest.get("train_sha256"),
        root / "dataset" / "train.jsonl",
    )
    add_hash_binding(
        "validation_payload_hash_binding",
        ds_manifest.get("validation_sha256"),
        root / "dataset" / "validation.jsonl",
    )
    add_hash_binding(
        "dataset_schema_hash_binding",
        ds_manifest.get("schema_sha256"),
        root / "dataset" / "schema.json",
    )
    add_hash_binding(
        "record_map_hash_binding",
        ds_manifest.get("record_map_sha256"),
        root / "provenance" / "record-map.jsonl",
    )
    add_hash_binding(
        "trainer_manifest_hash_binding",
        ds_manifest.get("trainer_upload_manifest_sha256"),
        root / "trainer-upload" / "manifest.json",
    )

    def add_byte_equality(name: str, first: Path, second: Path) -> None:
        try:
            equal = first.read_bytes() == second.read_bytes()
            detail = (
                f"byte-identical: {first} and {second}"
                if equal
                else f"byte mismatch: {first} and {second}"
            )
            add_check(name, equal, detail)
        except Exception as exc:
            add_check(name, False, str(exc))

    add_byte_equality(
        "trainer_train_byte_equality",
        root / "dataset" / "train.jsonl",
        root / "trainer-upload" / "train.jsonl",
    )
    add_byte_equality(
        "trainer_validation_byte_equality",
        root / "dataset" / "validation.jsonl",
        root / "trainer-upload" / "validation.jsonl",
    )
    add_byte_equality(
        "trainer_schema_byte_equality",
        root / "dataset" / "schema.json",
        root / "trainer-upload" / "schema.json",
    )
    try:
        trainer_manifest = json.loads(
            (root / "trainer-upload" / "manifest.json").read_text(encoding="utf-8")
        )
        trainer_errors: list[str] = []
        expected_trainer_values = {
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "train_sha256": sha256_file(root / "trainer-upload" / "train.jsonl"),
            "validation_sha256": sha256_file(
                root / "trainer-upload" / "validation.jsonl"
            ),
            "schema_sha256": sha256_file(root / "trainer-upload" / "schema.json"),
            "example_sha256": sha256_file(root / "trainer-upload" / "EXAMPLE.jsonl"),
        }
        for field, expected in expected_trainer_values.items():
            if trainer_manifest.get(field) != expected:
                trainer_errors.append(f"{field} mismatch")
        if trainer_manifest.get("model_visible_top_level_fields") != ds_manifest.get(
            "model_visible_top_level_fields"
        ):
            trainer_errors.append("model-visible field declaration mismatch")
        if trainer_manifest.get("recipe_id") != ds_manifest.get("recipe_id"):
            trainer_errors.append("recipe mismatch")
        if trainer_manifest.get("record_format") != ds_manifest.get("record_format"):
            trainer_errors.append("record format mismatch")
        add_check(
            "trainer_manifest_integrity",
            not trainer_errors,
            "; ".join(trainer_errors),
        )
    except Exception as exc:
        add_check("trainer_manifest_integrity", False, str(exc))

    source_hashes = {
        source.get("ingot_id"): source.get("ingot_hash")
        for source in sources
        if isinstance(source.get("ingot_id"), str)
    }
    decision_errors: list[str] = []
    decision_ids: list[str] = []
    source_ids_in_order = [
        source.get("ingot_id")
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("ingot_id"), str)
    ]
    if len(source_ids_in_order) != len(sources):
        decision_errors.append("every source ingot row must have a string ingot_id")
    if len(source_ids_in_order) != len(set(source_ids_in_order)):
        decision_errors.append("duplicate ingot_id in source ingot ledger")
    for index, decision in enumerate(decisions):
        decision_ingot_id = decision.get("ingot_id")
        if isinstance(decision_ingot_id, str):
            decision_ids.append(decision_ingot_id)
        else:
            decision_errors.append(f"decision[{index}] ingot_id is not a string")
        schema_errors = _schema_errors(decision, "curation-decision.v1.schema.json")
        if schema_errors:
            decision_errors.append(f"decision[{index}] schema: {'; '.join(schema_errors)}")
            continue
        receipt_hash = decision.get("receipt_hash")
        recomputed_receipt = sha256_json(
            {key: value for key, value in decision.items() if key != "receipt_hash"}
        )
        if receipt_hash != recomputed_receipt:
            decision_errors.append(f"decision[{index}] receipt_hash mismatch")
        if decision.get("product_spec_hash") != manifest.get("product_spec_hash"):
            decision_errors.append(f"decision[{index}] product_spec_hash mismatch")
        if decision.get("rubric_hash") != sha256_json(rubric):
            decision_errors.append(f"decision[{index}] rubric_hash mismatch")
        if source_hashes.get(decision_ingot_id) != decision.get("ingot_hash"):
            decision_errors.append(f"decision[{index}] ingot_hash mismatch")
    if len(decision_ids) != len(set(decision_ids)):
        decision_errors.append("duplicate ingot_id in curation decisions")
    if set(decision_ids) != set(source_hashes):
        decision_errors.append("curation decisions and source ingot ledger do not reconcile")
    add_check(
        "curation_decision_integrity",
        not decision_errors,
        "; ".join(decision_errors),
    )

    # Rights
    rights = json.loads((root / "provenance" / "rights.json").read_text(encoding="utf-8")) if (root / "provenance" / "rights.json").exists() else {}
    rights_status = rights.get("rights_status")
    declared_license = rights.get("license")
    rights_ok = (
        isinstance(declared_license, str)
        and bool(declared_license.strip())
        and declared_license != "UNSPECIFIED"
        and rights_status in {"fixture", "declared", "cleared"}
    )
    add_check(
        "rights_metadata_present",
        rights_ok,
        f"status={rights_status!r} license={declared_license!r}",
    )

    # Privacy: product_spec privacy_requirements presence; high-severity unresolved not accepted
    privacy_req = product_spec.get("privacy_requirements") or {}
    privacy_declared = bool(privacy_req)
    add_check(
        "privacy_requirements_declared",
        privacy_declared or fixture_only,
        "declared" if privacy_declared else "missing privacy_requirements",
    )

    # Eval pack integrity. The current grader is a declarative contract, not
    # executable code, so Hallmark names this check honestly and verifies the
    # complete case/answer/grader relationship without claiming a model eval ran.
    case_ids = [
        case.get("id")
        for case in eval_cases
        if isinstance(case, dict) and isinstance(case.get("id"), str) and case.get("id")
    ]
    answer_ids = [
        answer.get("id")
        for answer in eval_answers
        if isinstance(answer, dict)
        and isinstance(answer.get("id"), str)
        and answer.get("id")
    ]
    eval_integrity_errors: list[str] = []
    if len(case_ids) != len(eval_cases):
        eval_integrity_errors.append("every eval case must have a nonempty string id")
    if len(answer_ids) != len(eval_answers):
        eval_integrity_errors.append("every eval answer must have a nonempty string id")
    if len(case_ids) != len(set(case_ids)):
        eval_integrity_errors.append("eval case ids must be unique")
    if len(answer_ids) != len(set(answer_ids)):
        eval_integrity_errors.append("eval answer ids must be unique")
    if set(case_ids) != set(answer_ids):
        eval_integrity_errors.append("eval case and answer ids must match exactly")
    if any("label" not in answer for answer in eval_answers if isinstance(answer, dict)):
        eval_integrity_errors.append("every eval answer must contain a label")
    add_check(
        "eval_case_answer_integrity",
        not eval_integrity_errors and bool(eval_cases),
        "; ".join(eval_integrity_errors) or f"matched={len(case_ids)}",
    )
    add_check(
        "eval_manifest_case_count",
        isinstance(eval_manifest.get("case_count"), int)
        and eval_manifest.get("case_count") == len(eval_cases)
        and len(eval_cases) == len(eval_answers),
        (
            f"declared={eval_manifest.get('case_count')} "
            f"cases={len(eval_cases)} answers={len(eval_answers)}"
        ),
    )

    grader_path = root / "eval" / "graders" / "grader.json"
    if grader_path.exists():
        try:
            grader = json.loads(grader_path.read_text(encoding="utf-8"))
            grader_ok = (
                isinstance(grader, dict)
                and isinstance(grader.get("grader_id"), str)
                and bool(grader["grader_id"])
                and grader.get("grader_id") == eval_manifest.get("grader_id")
                and grader.get("type") == "deterministic"
            )
            add_check(
                "eval_grader_contract",
                grader_ok,
                (
                    f"manifest={eval_manifest.get('grader_id')!r} "
                    f"grader={grader.get('grader_id')!r} type={grader.get('type')!r}"
                    if isinstance(grader, dict)
                    else "grader must be a JSON object"
                ),
            )
        except Exception as exc:
            add_check("eval_grader_contract", False, str(exc))
    else:
        add_check("eval_grader_contract", False, "missing grader")

    # SHA256SUMS binds each payload file.  The manifest is checked separately
    # by its schema and the canonical product hash above.
    sums_ok = True
    sums_detail = ""
    if (root / "SHA256SUMS").exists():
        declared_sums: dict[str, str] = {}
        for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                digest, rel = line.split(None, 1)
            except ValueError:
                sums_ok = False
                break
            rel = rel.strip()
            relative = Path(rel)
            path = (root / relative).resolve()
            if (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or relative.is_absolute()
                or ".." in relative.parts
                or not path.is_relative_to(root)
                or path.is_symlink()
                or not path.is_file()
                or rel in declared_sums
            ):
                sums_ok = False
                break
            declared_sums[rel] = digest
        expected_sum_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.relative_to(root).as_posix()
            not in {"SHA256SUMS", "product-manifest.json"}
        }
        if set(declared_sums) != expected_sum_paths:
            sums_ok = False
            sums_detail = (
                f"missing={sorted(expected_sum_paths - set(declared_sums))} "
                f"extra={sorted(set(declared_sums) - expected_sum_paths)}"
            )
        elif sums_ok:
            for rel, digest in declared_sums.items():
                if sha256_file(root / rel) != digest:
                    sums_ok = False
                    sums_detail = f"digest mismatch: {rel}"
                    break
    else:
        sums_ok = False
        sums_detail = "missing SHA256SUMS"
    add_check("checksums", sums_ok, sums_detail)

    # Semantic re-review independent layer. A callback returning is not itself
    # approval: its explicit disposition controls the Hallmark outcome and the
    # exact result is retained in the report.
    reviewer = semantic_reviewer
    if reviewer is None:
        reviewer = fixture_semantic_reviewer if mode == "fixture" else production_semantic_reviewer
    semantic_results: list[dict[str, Any]] = []
    semantic_error: str | None = None
    for src in sources:
        try:
            # The current lot exposes source metadata rather than the original
            # ingot content. Keep that limitation visible; never turn a mere
            # callback return into an implicit ACCEPT.
            result = reviewer(src, rubric)
            if not isinstance(result, dict):
                raise ValueError("semantic reviewer must return an object")
            disposition = result.get("disposition")
            if disposition not in {"ACCEPT", "REJECT", "HOLD"}:
                raise ValueError(
                    "semantic reviewer disposition must be ACCEPT, REJECT, or HOLD"
                )
            # Prove the retained result is canonical-JSON serializable before it
            # becomes part of the signed-by-hash report body.
            sha256_json(result)
            captured = dict(result)
            reviewer_source_id = captured.get("source_ingot_id")
            actual_source_id = src.get("ingot_id")
            if reviewer_source_id is not None and reviewer_source_id != actual_source_id:
                raise ValueError("semantic reviewer returned a mismatched source_ingot_id")
            captured["source_ingot_id"] = actual_source_id
            semantic_results.append(captured)
        except Exception as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"
            semantic_results.append(
                {
                    "disposition": "HOLD",
                    "source_ingot_id": src.get("ingot_id"),
                    "reviewer_type": "error",
                    "blocked_reason": "semantic_reviewer_error",
                    "notes": semantic_error,
                }
            )
            break
    dispositions = [result["disposition"] for result in semantic_results]
    semantic_ok = bool(semantic_results) and all(
        disposition == "ACCEPT" for disposition in dispositions
    )
    add_check(
        "semantic_rereview",
        semantic_ok,
        semantic_error or f"reviewed={len(semantic_results)} dispositions={dispositions}",
    )
    raw_reviewed_source_ids = [
        result.get("source_ingot_id") for result in semantic_results
    ]
    reviewed_source_ids = [
        source_id
        for source_id in raw_reviewed_source_ids
        if isinstance(source_id, str)
    ]
    add_check(
        "semantic_review_attribution",
        len(reviewed_source_ids) == len(source_ids_in_order)
        and len(reviewed_source_ids) == len(set(reviewed_source_ids))
        and set(reviewed_source_ids) == set(source_ids_in_order),
        (
            f"sources={source_ids_in_order} "
            f"reviewed={raw_reviewed_source_ids}"
        ),
    )

    # The deterministic fixture reviewer has no authority over a non-fixture
    # product. It may help exercise the pipeline, but it cannot create a PASS
    # that looks like production review.
    add_check(
        "semantic_reviewer_scope",
        reviewer is not fixture_semantic_reviewer or fixture_only,
        "fixture reviewer is restricted to fixture_only products",
    )

    # Mode constraints
    if mode == "production" and (fixture_only or (reviewer is fixture_semantic_reviewer)):
        add_check("production_mode_rejects_fixture_reviewer", False, "fixture reviewer/product in production mode")
    else:
        add_check("production_mode_policy", True, mode)

    # Eval pack required
    add_check("eval_pack_present", (root / "eval" / "pack" / "cases.jsonl").exists())
    add_check("curation_decisions_present", bool(decisions))

    # Bind the report to a stable view of the exact lot. This detects ordinary
    # concurrent mutation across Hallmark's review window; Hallmark never
    # writes into the product directory itself.
    declared = manifest.get("product_hash")
    try:
        recomputed = sha256_product_lot(root)
        final_tree_hash = sha256_tree(root)
        add_check(
            "product_hash_binding",
            isinstance(declared, str) and len(declared) == 64 and declared == recomputed,
            f"declared={declared} recomputed={recomputed}",
        )
    except Exception as exc:
        recomputed = ""
        final_tree_hash = ""
        add_check("product_hash_binding", False, str(exc))
    add_check(
        "product_lot_stability",
        bool(initial_tree_hash)
        and bool(final_tree_hash)
        and initial_tree_hash == final_tree_hash,
        (
            f"before={initial_tree_hash or initial_hash_error} "
            f"after={final_tree_hash}"
        ),
    )

    # Status
    hard_fails = [c for c in checks if not c["ok"] and not c["name"].startswith("exists:reports")]
    status = "PASS"
    if any(not c["ok"] for c in checks):
        # HOLD if only semantic/privacy soft issues; FAIL for structural
        structural_names = {
            "product_manifest_schema",
            "product_hash_binding",
            "product_lot_stability",
            "checksums",
            "dataset_manifest_binding",
            "eval_manifest_binding",
            "product_spec_hash_binding",
            "curation_ledger_hash_binding",
            "source_ingot_ledger_hash_binding",
            "lineage_manifest_hash_binding",
            "rights_manifest_hash_binding",
            "train_payload_hash_binding",
            "validation_payload_hash_binding",
            "dataset_schema_hash_binding",
            "record_map_hash_binding",
            "trainer_manifest_hash_binding",
            "trainer_train_byte_equality",
            "trainer_validation_byte_equality",
            "trainer_schema_byte_equality",
            "trainer_manifest_integrity",
            "curation_decision_integrity",
            "dataset_eval_version_correspondence",
            "train_jsonl_parse",
            "validation_jsonl_parse",
            "decisions_jsonl_parse",
            "source_ingots_parse",
            "record_map_parse",
            "train_count_match",
            "validation_count_match",
            "record_map_integrity",
            "model_visible_schema",
            "included_have_decisions",
            "excluded_not_in_train",
            "source_ingot_resolvable",
            "eval_cases_loadable",
            "eval_answers_loadable",
            "no_split_leakage",
            "eval_case_answer_integrity",
            "eval_manifest_case_count",
            "eval_pack_present",
            "curation_decisions_present",
            "eval_grader_contract",
            "semantic_review_attribution",
            "semantic_reviewer_scope",
            "production_mode_rejects_fixture_reviewer",
        }
        if any(
            (not c["ok"])
            and (
                c["name"] in structural_names
                or c["name"].startswith("exists:")
                # An empty product is not a soft finding to be reviewed later.
                or c["name"].startswith("substance.")
            )
            for c in checks
        ):
            status = "FAIL"
        elif any(not c["ok"] for c in checks):
            status = "HOLD"
        else:
            status = "PASS"
    if any(result.get("disposition") == "REJECT" for result in semantic_results):
        status = "FAIL"

    report = {
        "schema_id": "goldtrace.hallmark.report.v1",
        "product_id": manifest.get("product_id") or product_spec.get("product_id") or "unknown",
        "product_version": manifest.get("product_version") or product_spec.get("product_version") or "unknown",
        "product_hash": recomputed or (declared if isinstance(declared, str) else ""),
        "status": status,
        "checks": checks,
        "discrepancies": discrepancies,
        "semantic_reviews": semantic_results,
        "reviewer": {
            "type": "fixture" if mode == "fixture" else "production",
            "mode": mode,
            "identity": getattr(reviewer, "__name__", "reviewer"),
        },
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fixture_only": fixture_only,
    }
    report["report_hash"] = sha256_json(report)
    report_errors = _schema_errors(report, "hallmark-report.v1.schema.json")
    if report_errors:
        raise ValueError(f"generated Hallmark report violates contract: {'; '.join(report_errors)}")
    return report
