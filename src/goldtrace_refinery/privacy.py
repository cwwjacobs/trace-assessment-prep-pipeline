from __future__ import annotations

import re
from typing import Any

# Deterministic secret/PII patterns only — no semantic judgment.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("generic_api_key", re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]


def scan_text(text: str) -> list[dict[str, str]]:
    """Return privacy-safe rule counts as individual findings.

    Findings deliberately contain neither matched text, byte spans, nor JSON
    paths.  Ingot and receipt provenance can be copied into a product pack, so
    even a short preview is an unsafe release-side credential fragment.
    """
    findings: list[dict[str, str]] = []
    for name, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            findings.append(
                {
                    "rule": name,
                }
            )
    return findings


def redact_text(text: str) -> tuple[str, list[dict[str, str]]]:
    findings = scan_text(text)
    out = text
    for name, pattern in _PATTERNS:
        out = pattern.sub(f"[REDACTED:{name}]", out)
    return out, findings


def scan_obj(obj: Any, path: str = "$") -> list[dict[str, str]]:
    del path  # retained for source compatibility; paths are intentionally private
    found: list[dict[str, str]] = []
    if isinstance(obj, str):
        found.extend(scan_text(obj))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(scan_obj(value))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(scan_obj(value))
    return found
