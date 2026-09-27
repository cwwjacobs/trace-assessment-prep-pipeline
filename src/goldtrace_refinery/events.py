from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import sha256_json


def parse_event_stream(events_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed event at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"event at line {line_no} is not an object")
            events.append(obj)
    return events


def verify_event_chain(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Lightweight chain check: sequence and optional prev-hash fields if present."""
    if not events:
        return {"ok": True, "count": 0, "breaks": []}

    breaks: list[str] = []
    prev_hash = None
    for index, event in enumerate(events):
        seq = event.get("sequence")
        if seq is not None and seq != index and seq != index + 1:
            # tolerate 0- or 1-based sequences; only flag regressions
            if isinstance(seq, int) and index > 0 and seq < (events[index - 1].get("sequence") or 0):
                breaks.append(f"sequence_regression_at_{index}")

        # common chain field names used in goldentrace events
        for key in ("prev_event_sha256", "previous_sha256", "parent_sha256"):
            if key in event and prev_hash is not None and event.get(key) not in (None, "", prev_hash):
                breaks.append(f"chain_break_at_{index}:{key}")
                break

        event_hash = event.get("event_sha256") or event.get("sha256")
        if isinstance(event_hash, str) and event_hash:
            prev_hash = event_hash
        else:
            prev_hash = sha256_json(event)

    return {"ok": len(breaks) == 0, "count": len(events), "breaks": breaks}


def normalize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical normalized event stream — shape only, no semantic labels."""
    normalized: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        etype = (
            event.get("event_type")
            or event.get("type")
            or event.get("kind")
            or "unknown"
        )
        normalized.append(
            {
                "index": index,
                "event_type": etype,
                "timestamp": event.get("timestamp") or event.get("ts") or event.get("created_at"),
                "payload": event,
            }
        )
    return normalized
