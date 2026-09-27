"""Privacy-treated factual projection of Mine interactive evidence.

This module follows only explicit references from Mine's verified mechanical
evidence index.  It does not select a dataset recipe or assign training value.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .paths import find_contract


SCHEMA_ID = "goldtrace.refinery.interactive_evidence.v1"
SCHEMA_NAME = "refinery-interactive-evidence.v1.schema.json"
ARTIFACT_NAME = "interactive-evidence.json"


def _contract() -> dict[str, Any]:
    # Resolve through the shared candidate search rather than a checkout-relative
    # path: the previous form reached three levels up into the superproject, which
    # exists in no installed release and in no clone of this component alone.
    return json.loads(find_contract(SCHEMA_NAME).read_text(encoding="utf-8"))


def validate_interactive_evidence(value: object) -> list[str]:
    validator = Draft202012Validator(_contract())
    return [
        error.message
        for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    ]


def _required_text(value: Mapping[str, Any], field: str, *, subject: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{subject} lacks {field}")
    return result


def _event_map(redacted_events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for wrapper in redacted_events:
        event = wrapper.get("payload")
        if not isinstance(event, dict):
            raise ValueError("redacted normalized event lacks its factual event payload")
        event_id = _required_text(event, "event_id", subject="redacted event")
        if event_id in mapped:
            raise ValueError(f"duplicate redacted event id: {event_id}")
        mapped[event_id] = event
    return mapped


def _contains_mine_cas_reference(value: object) -> bool:
    if isinstance(value, dict):
        if isinstance(value.get("cas_sha256"), str):
            return True
        return any(_contains_mine_cas_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_mine_cas_reference(item) for item in value)
    return False


def _treated(value: object, *, missing_reason: str = "source_field_unavailable") -> dict[str, Any]:
    if value is None:
        return {"available": False, "reason": missing_reason}
    if _contains_mine_cas_reference(value):
        return {
            "available": False,
            "reason": "privacy_treated_large_content_unavailable",
        }
    return {"available": True, "value": value}


def _event_payload(
    events: Mapping[str, dict[str, Any]],
    event_id: str,
) -> dict[str, Any]:
    event = events.get(event_id)
    if event is None:
        raise ValueError(f"interactive evidence references missing redacted event {event_id}")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"redacted event {event_id} lacks a payload object")
    return payload


def _message(
    events: Mapping[str, dict[str, Any]],
    event_id: str,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    payload = _event_payload(events, event_id)
    content: object = None
    for field in fields:
        if field in payload:
            content = payload[field]
            break
    return {"event_id": event_id, "content": _treated(content)}


def _tool_arguments(payload: Mapping[str, Any]) -> object:
    for field in ("tool_input", "arguments", "command"):
        if field in payload:
            return payload[field]
    return None


def _tool_result(payload: Mapping[str, Any]) -> object:
    for field in ("tool_response", "result", "aggregated_output", "output"):
        if field in payload:
            return payload[field]
    return None


def build_interactive_evidence(
    *,
    run: Mapping[str, Any],
    source_bundle_hash: str,
    source_manifest_hash: str,
    source_index: Mapping[str, Any],
    source_index_sha256: str,
    redacted_events: list[dict[str, Any]],
    redacted_stream_sha256: str,
    privacy_findings_count: int,
) -> dict[str, Any]:
    """Build a deterministic privacy-treated view using explicit index edges."""

    if run.get("classification") != "live_interactive":
        raise ValueError("interactive projection requires a live_interactive run")
    run_id = _required_text(run, "run_id", subject="interactive run")
    status = _required_text(run, "session_status", subject="interactive run")
    if status not in {"COMPLETED", "FAILED", "INTERRUPTED"}:
        raise ValueError("interactive run has an unsupported session status")
    if source_index.get("run_id") != run_id:
        raise ValueError("interactive index changes the Mine run identity")

    events = _event_map(redacted_events)
    indexed_calls = source_index.get("tool_calls")
    indexed_turns = source_index.get("turns")
    if not isinstance(indexed_calls, list) or not isinstance(indexed_turns, list):
        raise ValueError("interactive index lacks turn/tool arrays")
    calls_by_request: dict[str, dict[str, Any]] = {}
    for call in indexed_calls:
        if not isinstance(call, dict):
            raise ValueError("interactive index tool call is not an object")
        request_event_id = _required_text(
            call, "request_event_id", subject="interactive tool call"
        )
        if request_event_id in calls_by_request:
            raise ValueError("interactive index repeats a tool request")
        calls_by_request[request_event_id] = call

    projected_turns: list[dict[str, Any]] = []
    for indexed_turn in indexed_turns:
        if not isinstance(indexed_turn, dict):
            raise ValueError("interactive index turn is not an object")
        session_id = _required_text(indexed_turn, "session_id", subject="interactive turn")
        turn_id = _required_text(indexed_turn, "turn_id", subject="interactive turn")
        user_ids = indexed_turn.get("user_input_event_ids")
        agent_ids = indexed_turn.get("agent_message_event_ids")
        request_ids = indexed_turn.get("tool_request_event_ids")
        if not all(isinstance(value, list) for value in (user_ids, agent_ids, request_ids)):
            raise ValueError("interactive turn relationship arrays are malformed")

        projected_calls: list[dict[str, Any]] = []
        for ordinal, request_event_id in enumerate(request_ids, start=1):
            if not isinstance(request_event_id, str):
                raise ValueError("interactive turn request reference is malformed")
            call = calls_by_request.get(request_event_id)
            if call is None:
                raise ValueError("interactive turn references an unindexed tool request")
            result_event_id = call.get("result_event_id")
            request_payload = _event_payload(events, request_event_id)
            result_payload: Mapping[str, Any] = {}
            if isinstance(result_event_id, str):
                result_payload = _event_payload(events, result_event_id)
            elif result_event_id is not None:
                raise ValueError("interactive tool result reference is malformed")
            effects = call.get("effect_event_ids")
            subsequent = call.get("subsequent_agent_message_event_ids")
            if not isinstance(effects, list) or not isinstance(subsequent, list):
                raise ValueError("interactive tool relationship arrays are malformed")
            projected_calls.append(
                {
                    "ordinal": ordinal,
                    "native_id_field": _required_text(
                        call, "native_id_field", subject=request_event_id
                    ),
                    "native_call_id": _required_text(
                        call, "native_call_id", subject=request_event_id
                    ),
                    "kind": _required_text(call, "kind", subject=request_event_id),
                    "tool_name": _required_text(
                        call, "tool_name", subject=request_event_id
                    ),
                    "request_event_id": request_event_id,
                    "result_event_id": result_event_id,
                    "effect_event_ids": list(effects),
                    "subsequent_agent_message_event_ids": list(subsequent),
                    "state": _required_text(call, "state", subject=request_event_id),
                    "arguments": _treated(_tool_arguments(request_payload)),
                    "result": _treated(
                        _tool_result(result_payload),
                        missing_reason="tool_result_unavailable",
                    ),
                }
            )

        final_agent_ids: list[str]
        if projected_calls:
            final_agent_ids = list(
                projected_calls[-1]["subsequent_agent_message_event_ids"]
            )
            if any(event_id not in agent_ids for event_id in final_agent_ids):
                raise ValueError("tool call references an agent message outside its turn")
        else:
            final_agent_ids = list(agent_ids)

        projected_turns.append(
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "user_inputs": [
                    _message(events, event_id, ("prompt", "text"))
                    for event_id in user_ids
                ],
                "agent_messages": [
                    _message(
                        events,
                        event_id,
                        ("last_assistant_message", "text"),
                    )
                    for event_id in agent_ids
                ],
                "final_agent_message_event_ids": final_agent_ids,
                "tool_calls": projected_calls,
            }
        )

    event_stream = source_index.get("derivation")
    if not isinstance(event_stream, dict):
        raise ValueError("interactive index lacks event-stream derivation")
    value = {
        "schema_id": SCHEMA_ID,
        "source": {
            "run_id": run_id,
            "classification": "live_interactive",
            "source_bundle_hash": source_bundle_hash,
            "source_manifest_hash": source_manifest_hash,
            "source_index_path": "interactive-evidence-index.json",
            "source_index_sha256": source_index_sha256,
            "source_event_stream_head_sha256": _required_text(
                event_stream,
                "source_event_head_sha256",
                subject="interactive index derivation",
            ),
        },
        "mechanical_session_status": status,
        "evidence_gap_ids": list(run.get("evidence_gap_ids") or []),
        "privacy": {
            "content_source": "normalized_events.redacted.jsonl",
            "content_source_sha256": redacted_stream_sha256,
            "findings_count": privacy_findings_count,
            "large_content_policy": "do_not_resolve_raw_mine_cas",
        },
        "turns": projected_turns,
    }
    errors = validate_interactive_evidence(value)
    if errors:
        raise ValueError(f"interactive evidence contract invalid: {errors}")
    return value


__all__ = [
    "ARTIFACT_NAME",
    "SCHEMA_ID",
    "build_interactive_evidence",
    "validate_interactive_evidence",
]
