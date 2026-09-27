"""Extract training-format content from ingots for product-lot output.

Each recipe function receives an ingot dict and a runs_dir Path (the
mine/runtime/runs/ directory where sealed bundles live) and returns a
list of dict records ready for JSONL serialisation.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


# ── helpers ────────────────────────────────────────────────────────────────


def _resolve_artifact(runs_dir: Path, sha256: str) -> bytes:
    """Read a CAS artifact from a sealed bundle by SHA-256."""
    path = runs_dir.parent / "artifacts" / "sha256"  # not how bundles are stored
    # Bundles live as siblings of the runs_dir; each bundle has artifacts/sha256/XX/hash
    # The ingot's mine_run_id tells us the bundle directory name.
    return _read_blob(runs_dir, sha256)


def _read_blob(bundle_path: Path, sha256: str) -> bytes:
    """Read a CAS blob from a specific bundle directory."""
    blob = bundle_path / "artifacts" / "sha256" / sha256[:2] / sha256
    if blob.exists():
        return blob.read_bytes()
    return b""


def _resolve_bundle(runs_dir: Path, ingot: dict[str, Any]) -> Path | None:
    """Return the sealed bundle path for an ingot, if available."""
    mine_run_id = ingot.get("mine_run_id")
    if not mine_run_id:
        return None
    candidate = runs_dir / mine_run_id
    return candidate if candidate.is_dir() else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_contract(name: str) -> dict[str, Any]:
    from goldtrace_refinery.paths import find_contract

    return json.loads(find_contract(name).read_text(encoding="utf-8"))



def _load_interactive_evidence(
    ingot: dict[str, Any],
    ingot_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    results = ingot.get("deterministic_results")
    descriptor = results.get("interactive_evidence") if isinstance(results, dict) else None
    if descriptor is None:
        return None
    if not isinstance(descriptor, dict):
        raise ValueError("interactive evidence descriptor must be an object")
    if ingot_root is None:
        raise ValueError("interactive evidence requires the Refinery ingot directory")
    relative = descriptor.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("interactive evidence descriptor lacks a path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("interactive evidence path is not safely relative")
    root = Path(ingot_root).resolve(strict=True)
    path = (root / relative_path).resolve(strict=True)
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError("interactive evidence artifact is not a safe regular file")
    declared_hash = descriptor.get("sha256")
    if not isinstance(declared_hash, str) or _sha256_file(path) != declared_hash:
        raise ValueError("interactive evidence artifact hash differs from its ingot")
    value = json.loads(path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(
            _repo_contract("refinery-interactive-evidence.v1.schema.json")
        ).iter_errors(value),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ValueError(f"interactive evidence artifact violates contract: {errors[0].message}")
    source = value.get("source")
    privacy = value.get("privacy")
    if (
        not isinstance(source, dict)
        or source.get("run_id") != ingot.get("mine_run_id")
        or source.get("source_bundle_hash") != ingot.get("source_bundle_hash")
        or source.get("source_manifest_hash") != ingot.get("source_manifest_hash")
        or source.get("source_index_sha256") != descriptor.get("source_index_sha256")
        or not isinstance(privacy, dict)
        or privacy.get("content_source_sha256") != descriptor.get("privacy_source_sha256")
        or value.get("mechanical_session_status") != ingot.get("mechanical_run_status")
    ):
        raise ValueError("interactive evidence artifact provenance differs from its ingot")
    return value, descriptor


def _load_scenario_trace_evidence(
    ingot: dict[str, Any],
    ingot_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    results = ingot.get("deterministic_results")
    descriptor = (
        results.get("scenario_trace_evidence") if isinstance(results, dict) else None
    )
    if descriptor is None:
        return None
    if not isinstance(descriptor, dict):
        raise ValueError("scenario trace evidence descriptor must be an object")
    if ingot_root is None:
        raise ValueError("scenario trace evidence requires the Refinery ingot directory")
    relative = descriptor.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("scenario trace evidence descriptor lacks a path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("scenario trace evidence path is not safely relative")
    root = Path(ingot_root).resolve(strict=True)
    path = (root / relative_path).resolve(strict=True)
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError("scenario trace evidence artifact is not a safe regular file")
    declared_hash = descriptor.get("sha256")
    if not isinstance(declared_hash, str) or _sha256_file(path) != declared_hash:
        raise ValueError("scenario trace evidence artifact hash differs from its ingot")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("scenario trace evidence artifact must be a JSON object")
    from goldtrace_refinery.scenario_trace import verify_scenario_trace_evidence

    verify_scenario_trace_evidence(value, ingot=ingot)
    source = value.get("source")
    verifier = value.get("verifier")
    if (
        not isinstance(source, dict)
        or source.get("source_trace_file_sha256")
        != descriptor.get("source_trace_file_sha256")
        or source.get("source_trace_content_sha256")
        != descriptor.get("source_trace_content_sha256")
        or not isinstance(verifier, dict)
        or verifier.get("content_sha256") != descriptor.get("verifier_content_sha256")
    ):
        raise ValueError("scenario trace evidence descriptor differs from artifact")
    return value, descriptor


def _extract_scenario_trace_recovery(
    ingot: dict[str, Any],
    evidence: dict[str, Any],
    descriptor: dict[str, Any],
) -> list[dict[str, Any]]:
    scenario = evidence.get("scenario")
    trajectory = evidence.get("trajectory")
    verifier = evidence.get("verifier")
    source = evidence.get("source")
    run = evidence.get("run")
    if not all(isinstance(value, dict) for value in (scenario, trajectory, verifier, source, run)):
        raise ValueError("scenario trace evidence sections are malformed")
    assert isinstance(scenario, dict)
    assert isinstance(trajectory, dict)
    assert isinstance(verifier, dict)
    assert isinstance(source, dict)
    assert isinstance(run, dict)
    raw_steps = trajectory.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return []
    attempts: list[dict[str, Any]] = []
    for expected_step, raw in enumerate(raw_steps):
        if not isinstance(raw, dict) or raw.get("step") != expected_step:
            raise ValueError("scenario trace recovery steps are not contiguous")
        result = raw.get("environment_result")
        if not isinstance(result, dict):
            raise ValueError("scenario trace recovery step lacks grounded result")
        attempts.append(
            {
                "attempt": expected_step + 1,
                "observation": raw.get("observation_given_to_model"),
                "model_output": raw.get("model_output"),
                "action": raw.get("model_action"),
                "environment_result": result,
                "trajectory_label": raw.get("failure_or_correction"),
                "result": "success" if result.get("ok") is True else "failed",
            }
        )
    checks = verifier.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("scenario trace recovery record lacks verifier checks")
    verification = {
        "status": verifier.get("status"),
        "trace_quality_gate": verifier.get("trace_quality_gate"),
        "terminal_reason": trajectory.get("terminal_reason"),
        "passed_check_ids": [
            check.get("check_id")
            for check in checks
            if isinstance(check, dict) and check.get("status") == "PASS"
        ],
        "model_self_report_used_as_truth": verifier.get(
            "model_self_report_used_as_truth"
        ),
    }
    return [
        {
            "task": scenario.get("bounded_hypothesis"),
            "attempts": attempts,
            "verification": verification,
            "_provenance": {
                "source_bundle_hash": source.get("source_bundle_hash"),
                "source_manifest_hash": source.get("source_manifest_hash"),
                "source_trace_artifact_id": source.get("source_trace_artifact_id"),
                "source_trace_file_sha256": source.get("source_trace_file_sha256"),
                "source_trace_content_sha256": source.get(
                    "source_trace_content_sha256"
                ),
                "source_scenario_trace_evidence_sha256": descriptor.get("sha256"),
                "scenario_id": scenario.get("scenario_id"),
                "source_family": source.get("source_family"),
                "seed": run.get("seed"),
            },
        }
    ]


def _available(value: object) -> object | None:
    if not isinstance(value, dict) or value.get("available") is not True:
        return None
    return value.get("value")


def _content_text(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def _extract_interactive_tool_use(
    ingot: dict[str, Any],
    evidence: dict[str, Any],
    descriptor: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    turns = evidence.get("turns")
    if not isinstance(turns, list):
        return records
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        user_inputs = turn.get("user_inputs")
        agent_messages = turn.get("agent_messages")
        final_ids = turn.get("final_agent_message_event_ids")
        tool_calls = turn.get("tool_calls")
        if not all(
            isinstance(value, list)
            for value in (user_inputs, agent_messages, final_ids, tool_calls)
        ):
            continue
        if not user_inputs or not tool_calls or not final_ids:
            continue

        messages: list[dict[str, Any]] = []
        user_event_ids: list[str] = []
        unavailable = False
        for user_input in user_inputs:
            if not isinstance(user_input, dict):
                unavailable = True
                break
            content = _available(user_input.get("content"))
            event_id = user_input.get("event_id")
            if content is None or not isinstance(event_id, str):
                unavailable = True
                break
            messages.append({"role": "user", "content": _content_text(content)})
            user_event_ids.append(event_id)
        if unavailable:
            continue

        normalized_calls: list[dict[str, Any]] = []
        for expected_ordinal, call in enumerate(tool_calls, start=1):
            if not isinstance(call, dict) or call.get("ordinal") != expected_ordinal:
                unavailable = True
                break
            arguments = _available(call.get("arguments"))
            result = _available(call.get("result"))
            result_event_id = call.get("result_event_id")
            result_state = call.get("state")
            if (
                arguments is None
                or result is None
                or not isinstance(result_event_id, str)
                or result_state not in {"RESULT_OBSERVED", "SUCCEEDED", "FAILED"}
            ):
                unavailable = True
                break
            normalized_id = f"call_{expected_ordinal:04d}"
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": normalized_id,
                            "type": "function",
                            "function": {
                                "name": call["tool_name"],
                                "arguments": json.dumps(
                                    arguments,
                                    sort_keys=True,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": normalized_id,
                    "content": _content_text(result),
                }
            )
            normalized_calls.append(
                {
                    "normalized_call_id": normalized_id,
                    "native_id_field": call["native_id_field"],
                    "native_call_id": call["native_call_id"],
                    "request_event_id": call["request_event_id"],
                    "result_event_id": result_event_id,
                    "effect_event_ids": list(call.get("effect_event_ids") or []),
                    "result_state": result_state,
                }
            )
        if unavailable:
            continue

        agent_by_id = {
            item.get("event_id"): item
            for item in agent_messages
            if isinstance(item, dict) and isinstance(item.get("event_id"), str)
        }
        final_event_ids: list[str] = []
        for event_id in final_ids:
            item = agent_by_id.get(event_id)
            content = _available(item.get("content")) if isinstance(item, dict) else None
            if content is None:
                unavailable = True
                break
            messages.append({"role": "assistant", "content": _content_text(content)})
            final_event_ids.append(event_id)
        if unavailable:
            continue

        source = evidence["source"]
        records.append(
            {
                "messages": messages,
                "_provenance": {
                    "session_id": turn.get("session_id"),
                    "turn_id": turn.get("turn_id"),
                    "user_input_event_ids": user_event_ids,
                    "agent_message_event_ids": final_event_ids,
                    "request_event_ids": [
                        call["request_event_id"] for call in normalized_calls
                    ],
                    "result_event_ids": [
                        call["result_event_id"] for call in normalized_calls
                    ],
                    "effect_event_ids": [
                        event_id
                        for call in normalized_calls
                        for event_id in call["effect_event_ids"]
                    ],
                    "normalized_tool_calls": normalized_calls,
                    "source_interactive_evidence_sha256": descriptor["sha256"],
                    "source_index_sha256": source["source_index_sha256"],
                },
            }
        )
    return records


# ── event readers ──────────────────────────────────────────────────────────


def _read_events(bundle_path: Path) -> list[dict[str, Any]]:
    """Read all events from a sealed bundle's events.ndjson."""
    ndjson = bundle_path / "events" / "events.ndjson"
    if not ndjson.exists():
        return []
    events: list[dict[str, Any]] = []
    with ndjson.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def _extract_conversations(
    bundle_path: Path,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract all model request→response pairs as conversation records."""
    records: list[dict[str, Any]] = []

    # Pair model requests (messages) with action/response events
    # Model request events have body_sha256 pointing to the chat completions request
    # Action events have the model's parsed proposal and the raw_proposal_sha256

    # Collect model requests that contain messages
    requests: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for evt in events:
        if evt.get("event_type") != "model":
            continue
        name = evt.get("name", "")
        if "request" not in name:
            continue
        payload = evt.get("payload", {})
        if not isinstance(payload, dict):
            continue
        sha = payload.get("body_sha256", "")
        size = payload.get("body_size_bytes", 0)
        if not sha or size < 100:
            continue
        blob = _read_blob(bundle_path, sha)
        if not blob:
            continue
        try:
            body = json.loads(blob)
        except json.JSONDecodeError:
            continue
        msgs = body.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue
        requests.append((evt, body))

    # Match each request to the following action events
    for req_evt, req_body in requests:
        req_seq = req_evt.get("sequence", 0)
        # find action events that follow this request
        action_evts = [
            e
            for e in events
            if e.get("event_type") == "action"
            and e.get("sequence", 0) > req_seq
        ]
        if not action_evts:
            continue

        # Find the closest action after this request
        action = min(action_evts, key=lambda e: e.get("sequence", 0))
        action_payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
        proposal = action_payload.get("proposal")
        parse_error = action_payload.get("parse_error")
        finish_reason = action_payload.get("finish_reason", "")
        usage = action_payload.get("usage", {})

        # Build the SFT record: messages + assistant response
        messages = list(req_body["messages"])
        # Add the model's response as the final assistant message
        # Resolve raw_proposal_sha256 if available, otherwise use proposal dict
        raw_sha = action_payload.get("raw_proposal_sha256", "")
        assistant_content = ""
        if raw_sha:
            raw_blob = _read_blob(bundle_path, raw_sha)
            if raw_blob:
                assistant_content = raw_blob.decode("utf-8", errors="replace").strip()

        if not assistant_content and proposal:
            assistant_content = json.dumps(proposal)

        messages.append({"role": "assistant", "content": assistant_content})

        records.append({
            "messages": messages,
            "source_bundle": bundle_path.name,
            "mine_run_id": req_evt.get("run_id"),
            "sequence": req_seq,
            "finish_reason": finish_reason,
            "usage": usage,
            "parse_error": parse_error is not None,
            "proposal_type": (
                proposal.get("action", "unknown") if isinstance(proposal, dict) else "unknown"
            ),
        })

    return records


# ── recipe formatters ───────────────────────────────────────────────────────


def format_sft_chat(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """OpenAI chat-format SFT pairs.

    Output shape: { "messages": [{"role": ..., "content": ...}, ...] }
    """
    sft_pairs: list[dict[str, Any]] = []
    for rec in records:
        msgs = rec.get("messages", [])
        if len(msgs) < 2:
            continue
        sft_pairs.append({
            "messages": [
                {"role": m["role"], "content": m.get("content", "")}
                for m in msgs
            ],
            "metadata": {
                "source_bundle": rec.get("source_bundle"),
                "mine_run_id": rec.get("mine_run_id"),
                "finish_reason": rec.get("finish_reason"),
                "usage": rec.get("usage"),
            },
        })
    return sft_pairs


def format_tool_use(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Tool-use trajectories with structured tool_call / tool_result messages.

    Detects tool proposals from assistant text (JSON proposal payloads)
    and formats them as OpenAI-compatible tool_calls.
    """
    trajectories: list[dict[str, Any]] = []
    for rec in records:
        msgs = rec.get("messages", [])
        if len(msgs) < 2:
            continue

        # Check if any assistant message contains a tool proposal
        is_tool = rec.get("proposal_type") == "tool"
        if not is_tool:
            for m in msgs:
                if m.get("role") == "assistant":
                    content = m.get("content", "")
                    if isinstance(content, str) and '"action":"tool"' in content:
                        is_tool = True
                        break
        if not is_tool:
            continue

        # Rebuild messages: convert assistant text proposals to tool_calls format
        tool_msgs: list[dict[str, Any]] = []
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "assistant" and isinstance(content, str):
                # Try to parse JSON proposal and convert to tool_calls
                try:
                    proposal = json.loads(content)
                    if isinstance(proposal, dict) and proposal.get("action") == "tool":
                        tool_name = proposal.get("tool", "unknown")
                        arguments = proposal.get("arguments", {})
                        tool_msgs.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": f"call_{tool_name}_{len(tool_msgs):04d}",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(arguments),
                                },
                            }],
                        })
                        continue
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
            tool_msgs.append({"role": role, "content": content if isinstance(content, str) else str(content)})

        trajectories.append({
            "messages": tool_msgs,
            "metadata": {
                "source_bundle": rec.get("source_bundle"),
                "mine_run_id": rec.get("mine_run_id"),
                "proposal_type": rec.get("proposal_type"),
                "usage": rec.get("usage"),
            },
        })
    return trajectories


def format_dpo_pairs(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """DPO preference pairs — chosen (successful) vs rejected (failed/parse-error).

    Where a model produced a parse error then recovered, or where a task
    failed, the earlier bad response is "rejected" and a later successful
    response to the same or similar input is "chosen".
    """
    pairs: list[dict[str, Any]] = []
    # Group records by bundle (run)
    bundles: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        bid = rec.get("source_bundle", "unknown")
        bundles.setdefault(bid, []).append(rec)

    for bundle_recs in bundles.values():
        bundle_recs.sort(key=lambda r: r.get("sequence", 0))
        for i in range(len(bundle_recs) - 1):
            current = bundle_recs[i]
            later = bundle_recs[i + 1]
            if current.get("parse_error"):
                messages = current.get("messages", [])
                if len(messages) < 3:
                    continue
                prompt = messages[:-1]
                rejected = messages[-1].get("content", "")

                later_msgs = later.get("messages", [])
                if len(later_msgs) < 3:
                    continue
                chosen = later_msgs[-1].get("content", "")
                if chosen == rejected:
                    continue

                pairs.append({
                    "prompt": [{"role": m["role"], "content": m.get("content", "")} for m in prompt],
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": {
                        "source_bundle": rec.get("source_bundle"),
                        "rejected_sequence": current.get("sequence"),
                        "chosen_sequence": later.get("sequence"),
                    },
                })

    return pairs


def format_recovery(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recovery sequences — failure → diagnosis → correction → success."""
    recoveries: list[dict[str, Any]] = []
    bundles: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        bid = rec.get("source_bundle", "unknown")
        bundles.setdefault(bid, []).append(rec)

    for bundle_recs in bundles.values():
        bundle_recs.sort(key=lambda r: r.get("sequence", 0))
        for i in range(len(bundle_recs) - 1):
            current = bundle_recs[i]
            later = bundle_recs[i + 1]
            if not current.get("parse_error"):
                continue
            if not later.get("parse_error"):
                recovery_steps = [{
                    "attempt": i + 1,
                    "action": current.get("messages", [{}])[-1].get("content", ""),
                    "error": "parse_error" if current.get("parse_error") else "none",
                    "result": "failed",
                }, {
                    "attempt": i + 2,
                    "action": later.get("messages", [{}])[-1].get("content", ""),
                    "error": "none",
                    "result": "success",
                }]
                recoveries.append({
                    "task": "cyber_arena_investigation",
                    "source_bundle": current.get("source_bundle"),
                    "attempts": recovery_steps,
                    "verification": "Later proposal parsed successfully; model self-corrected.",
                })
                break  # one recovery sequence per bundle

    return recoveries


def format_eval_pairwise(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pairwise evaluation cases from conversation records.

    Each record becomes an eval case where the task is to compare the
    model's response against expected behaviour.
    """
    cases: list[dict[str, Any]] = []
    for rec in records:
        msgs = rec.get("messages", [])
        if len(msgs) < 3:
            continue
        # Prompt is everything up to the model response
        prompt = [{"role": m["role"], "content": m.get("content", "")[:200]} for m in msgs[:-1]]
        response = msgs[-1].get("content", "")[:200]
        case = {
            "id": f"eval-{rec.get('source_bundle','?')}-{rec.get('sequence','?')}",
            "source_id": rec.get("mine_run_id", "?"),
            "task": "Evaluate whether the model's tool proposal is correct.",
            "input": {
                "prompt": prompt,
                "candidate_response": response,
                "expected_behaviour": "valid_proposal_or_appropriate_error",
            },
            "split_group_id": rec.get("mine_run_id", "?"),
        }
        cases.append(case)
    return cases


# ── main entry ─────────────────────────────────────────────────────────────


def extract_content(
    ingot: dict[str, Any],
    recipe_id: str,
    runs_dir: Path,
    *,
    ingot_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Extract training-format records from an ingot.

    Args:
        ingot: loaded ingot dict (must have mine_run_id and source_class).
        recipe_id: one of 'sft-chat-v1', 'tool-use-v1', 'dpo-preference-v1',
                   'recovery-v1', 'eval-pairwise-v1'.
        runs_dir: path to mine/runtime/runs/ directory.

    Returns:
        list of training record dicts for the selected recipe.
    """
    source_class = ingot.get("source_class", "")
    mine_run_id = ingot.get("mine_run_id")

    records: list[dict[str, Any]] = []

    scenario_trace = _load_scenario_trace_evidence(ingot, ingot_root)
    if scenario_trace is not None:
        # Causal recovery data is produced only from Refinery's bounded derivative.
        # Never fall through to the sealed source event stream, which contains
        # controller-only world truth and hidden-event evidence.
        if recipe_id != "recovery-v1":
            return []
        evidence, descriptor = scenario_trace
        return _extract_scenario_trace_recovery(ingot, evidence, descriptor)

    interactive = _load_interactive_evidence(ingot, ingot_root)
    if interactive is not None:
        # Interactive model-visible content must come only from Refinery's
        # privacy-treated derivative. Never fall through to raw Mine CAS.
        if recipe_id != "tool-use-v1":
            return []
        evidence, descriptor = interactive
        return _extract_interactive_tool_use(ingot, evidence, descriptor)

    if source_class == "goldtrace_mine_bundle" and mine_run_id:
        bundle_path = runs_dir / mine_run_id
        if bundle_path.is_dir():
            events = _read_events(bundle_path)
            records = _extract_conversations(bundle_path, events)

    elif source_class == "rune_capture":
        seed_text = ingot.get("trajectory_summary", {}).get("seed_text", "")
        provenance = ingot.get("trajectory_summary", {}).get("provenance_statement", "")
        if seed_text:
            records.append({
                "messages": [
                    {"role": "user", "content": seed_text},
                    {"role": "assistant", "content": provenance},
                ],
                "source_bundle": ingot.get("source_bundle_hash", ""),
                "mine_run_id": mine_run_id,
                "sequence": 0,
                "finish_reason": "stop",
                "usage": {},
                "parse_error": False,
                "proposal_type": "seed",
            })

    # Dispatch to formatter
    formatters = {
        "sft-chat-v1": format_sft_chat,
        "tool-use-v1": format_tool_use,
        "dpo-preference-v1": format_dpo_pairs,
        "recovery-v1": format_recovery,
        "eval-pairwise-v1": format_eval_pairwise,
    }

    formatter = formatters.get(recipe_id)
    if formatter:
        return formatter(records)

    return records
