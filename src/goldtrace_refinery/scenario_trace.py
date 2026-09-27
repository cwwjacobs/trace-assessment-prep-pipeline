"""Refinery projection for verified controller-owned causal scenario traces.

The Labyrinth run.v8 verifier remains the authority for the sealed source. This
module re-opens only the exact CAS-bound proof artifacts already enumerated by
that verified run, replays the additive Scenario Assay contract, and emits a
privacy-bounded derivative suitable for deterministic recovery-data production.

World-state snapshots, hidden exogenous event identities, hidden paths, and
state deltas remain in the sealed evidence bundle. The model-facing derivative
contains only observations actually given to the target, observable outputs,
actions, grounded environment results, preserved failure/correction labels, and
independent verifier dispositions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .paths import ensure_labyrinth_importable, find_contract


SCHEMA_ID = "gtdataworks.refinery.scenario-trace-evidence.v1"
ARTIFACT_NAME = "scenario-trace-evidence.json"
ROUTING_SCHEMA_ID = "gtdataworks.processor-routing-receipt.v1"
ROUTING_ARTIFACT_NAME = "processor-routing-receipt.json"
PROJECTION_VERSION = "1.0.0"
PROJECTION_POLICY_ID = "gtdataworks.scenario-trace-recovery-projection.v1"
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9._-]+$")
_ALLOWED_PROOF_FILES = frozenset(
    {
        "capability-record.json",
        "environment-spec.json",
        "manifest.json",
        "model-run-spec.json",
        "model-target.json",
        "native-trace-envelope.json",
        "scenario-run-receipt.json",
        "scenario-spec.json",
        "verifier-receipt.json",
    }
)

CANONICAL_MODEL_ROUTES = (
    {
        "source_family": "CLI_CODING",
        "processor": "GTDataworks-Refinery",
        "output_class": "CLI_INGOT",
    },
    {
        "source_family": "LOCAL_MODEL",
        "processor": "GTDataworks-Labyrinth",
        "output_class": "LM_INGOT",
    },
    {
        "source_family": "CLOUD_MODEL",
        "processor": "GTDataworks-Cinderfield",
        "output_class": "CM_INGOT",
    },
)
CONTROL_ROUTE = {
    "source_family": "CONTROL_POLICY",
    "processor": "GTDataworks-Refinery",
    "output_class": "CONTROL_REFERENCE_INGOT",
    "route_class": "additive_control_baseline",
    "gold_eligible": False,
    "sale_ready": False,
}


class ScenarioTraceEvidenceError(ValueError):
    """Verified scenario-trace evidence cannot be projected without ambiguity."""


@dataclass(frozen=True)
class ScenarioTraceEvidenceResult:
    evidence: dict[str, Any]
    routing_receipt: dict[str, Any]
    source_family: str
    scenario_id: str
    scenario_version: str
    trace_artifact_id: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ScenarioTraceEvidenceError("value is not canonical JSON") from exc


def _sha256_json(value: object) -> str:
    return _sha256(_canonical_json(value))


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ScenarioTraceEvidenceError(message)


def _load_schema(name: str) -> dict[str, Any]:
    try:
        value = json.loads(find_contract(name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioTraceEvidenceError(f"scenario-trace contract is unavailable: {name}") from exc
    _require(isinstance(value, dict), f"scenario-trace contract is not an object: {name}")
    assert isinstance(value, dict)
    return value


def _validate_schema(value: object, *, contract: str, subject: str) -> None:
    errors = sorted(
        Draft202012Validator(_load_schema(contract)).iter_errors(value),
        key=lambda item: list(item.path),
    )
    if errors:
        location = "/".join(str(part) for part in errors[0].path) or "<root>"
        raise ScenarioTraceEvidenceError(
            f"{subject} violates {contract} at {location}: {errors[0].message}"
        )


def _read_regular_file(path: Path, *, expected_size: int, expected_sha256: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ScenarioTraceEvidenceError(f"scenario-trace CAS artifact is unavailable: {path}") from exc
    _require(
        stat.S_ISREG(before.st_mode)
        and not path.is_symlink()
        and before.st_nlink == 1
        and before.st_size == expected_size,
        "scenario-trace CAS artifact is not a stable single-link regular file",
    )
    data = path.read_bytes()
    after = path.lstat()
    _require(
        (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        and len(data) == expected_size
        and _sha256(data) == expected_sha256,
        "scenario-trace CAS artifact changed or differs from its descriptor",
    )
    return data


def _proof_bytes(bundle_root: Path, run: Mapping[str, object]) -> dict[str, bytes]:
    runtime = run.get("runtime")
    detail = runtime.get("scenario_trace") if isinstance(runtime, dict) else None
    descriptors = detail.get("proof_files") if isinstance(detail, dict) else None
    _require(isinstance(descriptors, list), "scenario-trace run lacks proof-file descriptors")
    proof: dict[str, bytes] = {}
    assert isinstance(descriptors, list)
    for descriptor in descriptors:
        _require(
            isinstance(descriptor, dict)
            and set(descriptor) == {"path", "size_bytes", "sha256", "role"}
            and descriptor.get("path") in _ALLOWED_PROOF_FILES
            and isinstance(descriptor.get("size_bytes"), int)
            and not isinstance(descriptor.get("size_bytes"), bool)
            and int(descriptor["size_bytes"]) > 0
            and isinstance(descriptor.get("sha256"), str)
            and _DIGEST.fullmatch(str(descriptor["sha256"])) is not None,
            "scenario-trace proof-file descriptor is malformed",
        )
        name = str(descriptor["path"])
        _require(
            _SAFE_RELATIVE.fullmatch(name) is not None and name not in proof,
            "scenario-trace proof-file descriptor is unsafe or duplicated",
        )
        digest = str(descriptor["sha256"])
        path = bundle_root / "artifacts" / "sha256" / digest[:2] / digest
        proof[name] = _read_regular_file(
            path,
            expected_size=int(descriptor["size_bytes"]),
            expected_sha256=digest,
        )
    _require(set(proof) == _ALLOWED_PROOF_FILES, "scenario-trace proof exact file set differs")
    return proof


def _projection_body(
    *,
    proof: object,
    run: Mapping[str, object],
    source_bundle_hash: str,
    source_manifest_hash: str,
    scenario_trace_index_sha256: str,
    import_package_sha256: str,
) -> dict[str, Any]:
    # ``proof`` is a ScenarioTraceProof imported lazily from Labyrinth.
    documents = getattr(proof, "documents")
    trace = getattr(proof, "trace")
    verifier = getattr(proof, "verifier")
    file_sha256 = getattr(proof, "file_sha256")
    scenario = documents["scenario-spec.json"]
    target = documents["model-target.json"]
    run_spec = documents["model-run-spec.json"]
    run_receipt = documents["scenario-run-receipt.json"]

    steps: list[dict[str, Any]] = []
    for raw in trace["step_records"]:
        _require(isinstance(raw, dict), "scenario trace step is malformed")
        steps.append(
            {
                "step": raw["step"],
                "observation_given_to_model": copy.deepcopy(
                    raw["observation_given_to_model"]
                ),
                "model_output": copy.deepcopy(raw["model_output"]),
                "model_action": copy.deepcopy(raw["model_action"]),
                "environment_result": copy.deepcopy(raw["environment_result"]),
                "failure_or_correction": raw["failure_or_correction"],
            }
        )

    checks = [
        {
            "check_id": check["check_id"],
            "category": check["category"],
            "status": check["status"],
            "description": check["description"],
            "independent_of_model_self_report": check[
                "independent_of_model_self_report"
            ],
        }
        for check in verifier["checks"]
    ]
    source_family = str(trace["source_family"])
    body: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "schema_version": PROJECTION_VERSION,
        "projection_policy_id": PROJECTION_POLICY_ID,
        "source": {
            "labyrinth_run_id": run["run_id"],
            "source_bundle_hash": source_bundle_hash,
            "source_manifest_hash": source_manifest_hash,
            "classification": "scenario_trace_import",
            "seal_status": run["seal_status"],
            "source_trace_artifact_id": trace["artifact_id"],
            "source_trace_file_sha256": file_sha256[
                "native-trace-envelope.json"
            ],
            "source_trace_content_sha256": trace["content_hash"]["value"],
            "proof_manifest_sha256": file_sha256["manifest.json"],
            "scenario_trace_index_sha256": scenario_trace_index_sha256,
            "import_package_sha256": import_package_sha256,
            "source_family": source_family,
        },
        "scenario": {
            "artifact_id": scenario["artifact_id"],
            "scenario_id": scenario["scenario_id"],
            "schema_id": scenario["schema_id"],
            "schema_version": scenario["schema_version"],
            "bounded_hypothesis": scenario["bounded_hypothesis"],
            "max_steps": scenario["max_steps"],
            "action_budget": scenario["action_budget"],
        },
        "target": {
            "artifact_id": target["artifact_id"],
            "target_id": target["target_id"],
            "model_name": target["model_name"],
            "model_version": target["model_version"],
            "adapter_id": target["adapter_id"],
            "adapter_version": target["adapter_version"],
            "source_family": target["source_family"],
        },
        "run": {
            "artifact_id": run_spec["artifact_id"],
            "run_id": run_spec["run_id"],
            "seed": run_spec["seed"],
            "status": run["import_status"],
            "source_scenario_run_status": run_receipt["status"],
            "terminal_reason": run_receipt["terminal_reason"],
            "steps_executed": run_receipt["steps_executed"],
            "actions_executed": run_receipt["actions_executed"],
        },
        "trajectory": {
            "trace_id": trace["trace_id"],
            "status": trace["status"],
            "terminal_reason": trace["terminal_reason"],
            "failure_and_correction_count": trace[
                "failure_and_correction_count"
            ],
            "steps": steps,
        },
        "verifier": {
            "artifact_id": verifier["artifact_id"],
            "file_sha256": file_sha256["verifier-receipt.json"],
            "content_sha256": verifier["content_hash"]["value"],
            "status": verifier["status"],
            "trace_quality_gate": verifier["trace_quality_gate"],
            "model_self_report_used_as_truth": verifier[
                "model_self_report_used_as_truth"
            ],
            "required_check_count": verifier["required_check_count"],
            "passed_check_count": verifier["passed_check_count"],
            "checks": checks,
        },
        "exclusions": {
            "world_truth_snapshots": "sealed_bundle_only",
            "hidden_exogenous_event_identities": "sealed_bundle_only",
            "hidden_state_paths": "sealed_bundle_only",
            "state_deltas": "sealed_bundle_only",
            "hidden_chain_of_thought": "not_required_and_not_captured",
            "model_facing_content_source": (
                "observation_given_to_model_model_output_model_action_"
                "grounded_environment_result_only"
            ),
        },
        "training_admission": {
            "evaluated": False,
            "status": "NOT_EVALUATED",
        },
        "gold_status": "NOT_GOLD",
    }
    body["evidence_sha256"] = _sha256_json(body)
    return body


def _assert_projection_excludes_hidden_surfaces(value: object) -> None:
    forbidden_keys = {
        "world_state",
        "world_state_before_ref",
        "world_state_after_ref",
        "state_delta",
        "hidden_event_refs",
        "event_schedule",
        "interventions",
    }

    def walk(item: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in forbidden_keys:
                    raise ScenarioTraceEvidenceError(
                        "scenario-trace model-facing derivative contains a hidden surface: "
                        + "/".join((*path, key))
                    )
                walk(child, (*path, key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, (*path, str(index)))

    trajectory = value.get("trajectory") if isinstance(value, dict) else None
    walk(trajectory)


def build_scenario_trace_evidence(
    *,
    bundle_root: Path,
    run: Mapping[str, object],
    source_bundle_hash: str,
    source_manifest_hash: str,
) -> ScenarioTraceEvidenceResult:
    _require(
        run.get("schema_version") == "goldentrace.run.v8"
        and run.get("classification") == "scenario_trace_import"
        and run.get("import_status") == "COMPLETED"
        and run.get("verification_status") == "VERIFIED_COMPLETE"
        and run.get("seal_status") == "SEALED_COMPLETE"
        and run.get("evidence_gap_ids") == [],
        "source is not a complete verified scenario-trace run.v8 import",
    )
    index_descriptor = run.get("scenario_trace_index")
    package_descriptor = run.get("scenario_trace_import_package")
    _require(
        isinstance(index_descriptor, dict)
        and index_descriptor.get("path") == "scenario-trace-index.json"
        and isinstance(index_descriptor.get("sha256"), str)
        and _DIGEST.fullmatch(str(index_descriptor["sha256"])) is not None
        and isinstance(package_descriptor, dict)
        and package_descriptor.get("path") == "scenario-trace-import-package.json"
        and isinstance(package_descriptor.get("sha256"), str)
        and _DIGEST.fullmatch(str(package_descriptor["sha256"])) is not None,
        "scenario-trace index/import-package descriptors are malformed",
    )
    index_path = bundle_root / "scenario-trace-index.json"
    package_path = bundle_root / "scenario-trace-import-package.json"
    _require(
        _sha256(index_path.read_bytes()) == index_descriptor["sha256"]
        and _sha256(package_path.read_bytes()) == package_descriptor["sha256"],
        "scenario-trace index/import-package bytes differ from run descriptors",
    )

    ensure_labyrinth_importable()
    try:
        from goldentrace.scenario_trace_contract import (
            inspect_scenario_trace_bytes,
        )
    except Exception as exc:  # pragma: no cover - installation/configuration failure
        raise ScenarioTraceEvidenceError(
            "Labyrinth scenario-trace contract is unavailable"
        ) from exc

    proof = inspect_scenario_trace_bytes(_proof_bytes(bundle_root, run))
    body = _projection_body(
        proof=proof,
        run=run,
        source_bundle_hash=source_bundle_hash,
        source_manifest_hash=source_manifest_hash,
        scenario_trace_index_sha256=str(index_descriptor["sha256"]),
        import_package_sha256=str(package_descriptor["sha256"]),
    )
    _assert_projection_excludes_hidden_surfaces(body)
    _validate_schema(
        body,
        contract="refinery-scenario-trace-evidence.v1.schema.json",
        subject="scenario-trace evidence",
    )
    source_family = str(body["source"]["source_family"])
    routing = build_processor_routing_receipt(
        source_family=source_family,
        source_bundle_hash=source_bundle_hash,
        source_manifest_hash=source_manifest_hash,
        labyrinth_run_id=str(run["run_id"]),
        trace_artifact_id=str(body["source"]["source_trace_artifact_id"]),
    )
    return ScenarioTraceEvidenceResult(
        evidence=body,
        routing_receipt=routing,
        source_family=source_family,
        scenario_id=str(body["scenario"]["scenario_id"]),
        scenario_version=str(body["scenario"]["schema_version"]),
        trace_artifact_id=str(body["source"]["source_trace_artifact_id"]),
    )


def build_processor_routing_receipt(
    *,
    source_family: str,
    source_bundle_hash: str,
    source_manifest_hash: str,
    labyrinth_run_id: str,
    trace_artifact_id: str,
) -> dict[str, Any]:
    routes = {item["source_family"]: dict(item) for item in CANONICAL_MODEL_ROUTES}
    routes["CONTROL_POLICY"] = dict(CONTROL_ROUTE)
    _require(source_family in routes, f"unsupported scenario-trace source family: {source_family}")
    selected = routes[source_family]
    receipt: dict[str, Any] = {
        "schema_id": ROUTING_SCHEMA_ID,
        "schema_version": "1.0.0",
        "source": {
            "source_family": source_family,
            "source_bundle_hash": source_bundle_hash,
            "source_manifest_hash": source_manifest_hash,
            "labyrinth_run_id": labyrinth_run_id,
            "trace_artifact_id": trace_artifact_id,
        },
        "canonical_model_routes": [dict(item) for item in CANONICAL_MODEL_ROUTES],
        "selected_route": selected,
        "existing_canon_modified": False,
        "crucible_used": False,
        "claim_boundary": (
            "CONTROL_POLICY is an additive deterministic baseline route to a "
            "control/reference ingot. It is not relabeled as CLI, local-model, "
            "or cloud-model output and is never Gold or sale-ready by routing alone."
            if source_family == "CONTROL_POLICY"
            else "The selected route preserves the established GTDataworks model-source canon."
        ),
    }
    receipt["receipt_sha256"] = _sha256_json(receipt)
    _validate_schema(
        receipt,
        contract="processor-routing-receipt.v1.schema.json",
        subject="processor routing receipt",
    )
    return receipt


def verify_scenario_trace_evidence(
    evidence: Mapping[str, object],
    *,
    ingot: Mapping[str, object] | None = None,
) -> None:
    _validate_schema(
        evidence,
        contract="refinery-scenario-trace-evidence.v1.schema.json",
        subject="scenario-trace evidence",
    )
    claimed = evidence.get("evidence_sha256")
    _require(isinstance(claimed, str) and _DIGEST.fullmatch(claimed) is not None, "scenario evidence hash is malformed")
    body = dict(evidence)
    body.pop("evidence_sha256", None)
    _require(_sha256_json(body) == claimed, "scenario evidence hash integrity failed")
    _assert_projection_excludes_hidden_surfaces(evidence)
    if ingot is not None:
        source = evidence.get("source")
        run = evidence.get("run")
        _require(
            isinstance(source, dict)
            and isinstance(run, dict)
            and source.get("source_bundle_hash") == ingot.get("source_bundle_hash")
            and source.get("source_manifest_hash") == ingot.get("source_manifest_hash")
            and source.get("labyrinth_run_id") == ingot.get("mine_run_id")
            and run.get("status") == ingot.get("mechanical_run_status"),
            "scenario evidence provenance differs from ingot",
        )


__all__ = [
    "ARTIFACT_NAME",
    "CANONICAL_MODEL_ROUTES",
    "CONTROL_ROUTE",
    "PROJECTION_POLICY_ID",
    "ROUTING_ARTIFACT_NAME",
    "ROUTING_SCHEMA_ID",
    "SCHEMA_ID",
    "ScenarioTraceEvidenceError",
    "ScenarioTraceEvidenceResult",
    "build_processor_routing_receipt",
    "build_scenario_trace_evidence",
    "verify_scenario_trace_evidence",
]
