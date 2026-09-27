#!/usr/bin/env python3
"""
GTDataworks Selective Token-Loss Masking
=======================================
Formats agent traces for frontier model fine-tuning with exact loss masking:
  - System prompts: loss_weight = 0.0 (masked)
  - Tool execution returns (stdout/stderr/compiler outputs): loss_weight = 0.0 (masked)
  - Model Chain-of-Thought reasoning (provider_reasoning): loss_weight = 1.0 (TRAINABLE)
  - Model tool-call actions: loss_weight = 1.0 (TRAINABLE)
  - Model final answers: loss_weight = 1.0 (TRAINABLE)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MaskedMessage:
    role: str                       # "system" | "user" | "assistant" | "tool"
    content: str
    loss_weight: float              # 0.0 for masked, 1.0 for trained
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


def format_trace_with_loss_mask(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform raw Labyrinth events into loss-masked training messages."""
    messages: List[MaskedMessage] = []

    for ev in events:
        ev_type = ev.get("type") or ev.get("name")
        payload = ev.get("payload") or ev.get("data") or {}

        # 1. System initialization (Masked)
        if ev_type in ["scenario.init", "system.prompt"]:
            messages.append(MaskedMessage(
                role="system",
                content=payload.get("instruction") or payload.get("text") or "",
                loss_weight=0.0
            ))

        # 2. User prompt (Masked)
        elif ev_type in ["user.input", "task.start"]:
            messages.append(MaskedMessage(
                role="user",
                content=payload.get("prompt") or payload.get("text") or "",
                loss_weight=0.0
            ))

        # 3. Model turn (TRAINABLE)
        elif ev_type in ["model.response", "turn.assistant"]:
            reasoning = payload.get("reasoning") or payload.get("provider_reasoning")
            content = payload.get("content") or ""
            tool_calls = payload.get("tool_calls") or []

            messages.append(MaskedMessage(
                role="assistant",
                content=content,
                reasoning_content=reasoning,
                tool_calls=tool_calls,
                loss_weight=1.0  # Train on reasoning and tool decisions
            ))

        # 4. Tool execution return (Masked - never train model to hallucinate tool output)
        elif ev_type in ["tool.result", "broker.result", "exec.result"]:
            tool_output = payload.get("output") or payload.get("result") or ""
            if isinstance(tool_output, (dict, list)):
                tool_output = json.dumps(tool_output)

            messages.append(MaskedMessage(
                role="tool",
                content=str(tool_output),
                loss_weight=0.0  # Masked!
            ))

    # Convert dataclasses to standard JSON format with loss weights
    formatted = []
    for msg in messages:
        entry: Dict[str, Any] = {
            "role": msg.role,
            "content": msg.content,
            "loss_weight": msg.loss_weight,
        }
        if msg.reasoning_content:
            entry["reasoning_content"] = msg.reasoning_content
        if msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls
        formatted.append(entry)

    return formatted


def export_loss_masked_jsonl(input_traces: List[List[Dict[str, Any]]], output_file: Path) -> int:
    """Export training traces into loss-weighted JSONL format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_file.open("w", encoding="utf-8") as f:
        for trace in input_traces:
            masked_msgs = format_trace_with_loss_mask(trace)
            if masked_msgs:
                row = {"messages": masked_msgs}
                f.write(json.dumps(row) + "\n")
                count += 1
    return count
