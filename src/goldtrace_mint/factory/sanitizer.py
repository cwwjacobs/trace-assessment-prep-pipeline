"""Gold Trace Mint — Privacy Sanitizer & Legal Compliance Auditor.

This module implements:
1. Deterministic PII and secret scrubbing with auditable replacement tokens:
   - Emails -> [REDACTED_EMAIL]
   - Phone numbers -> [REDACTED_PHONE]
   - IP addresses (IPv4 & IPv6) -> [REDACTED_IP_ADDRESS]
   - API keys and tokens -> [REDACTED_API_KEY]
   - Credentials, passwords, private keys -> [REDACTED_CREDENTIAL], [REDACTED_PRIVATE_KEY], [REDACTED_PASSWORD]
   - Personal names -> [REDACTED_NAME]
   - SSNs, credit cards, local file paths -> [REDACTED_SSN], [REDACTED_CREDIT_CARD], [REDACTED_PATH]
   - Operator-registered sensitive phrases -> [REDACTED_ALIAS], [REDACTED_ENTITY], [SYNTHETIC_SIGNAL]
   - Path redaction converting the operator's own home directory to /workspace/...
     (configurable via `workspace_home`; every other user's home -> [REDACTED_PATH])
2. SensitivePhraseRegistry for operator-supplied aliases, codenames, and other sensitive phrases.
   No phrases ship built in: each deployment registers its own.
3. Stand-alone batch sanitization CLI function `sanitize_file(input_path, output_path)`.
4. Strict legal license verification:
   - Allows only permissive licenses (MIT, Apache 2.0, BSD, Unlicense, CC0, ISC, 0BSD)
   - Rejects GPL copyleft (GPL, AGPL, LGPL, SSPL, EUPL, MPL)
   - Rejects proprietary / commercial / restricted / unlicensed sources
5. Cryptographic, fail-closed audit receipts with SHA-256 integrity verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "DEFAULT_REDACTION_TOKENS",
    "ComplianceAuditor",
    "LicenseCategory",
    "LicenseVerificationResult",
    "LicenseVerifier",
    "PIICategory",
    "PIIFinding",
    "PIISanitizer",
    "RepositoryLicenseReport",
    "SanitizationReceipt",
    "SensitivePhraseEntry",
    "SensitivePhraseRegistry",
    "canonical_json",
    "main",
    "redact_pii",
    "sanitize_file",
    "scan_pii",
    "sha256_bytes",
    "sha256_json",
    "sha256_text",
    "verify_license",
    "verify_repository_license",
]


# ============================================================================
# HASHING AND AUDIT UTILITIES
# ============================================================================

def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hexadecimal digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Compute SHA-256 hexadecimal digest of text encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Serialize object to canonical, deterministic JSON bytes."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def sha256_json(obj: Any) -> str:
    """Compute SHA-256 hexadecimal digest of canonical JSON representation."""
    return sha256_bytes(canonical_json(obj))


# ============================================================================
# PII CATEGORIES & TOKENS
# ============================================================================

class PIICategory(str, Enum):
    """Categories of PII, secrets, and sensitive data."""
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    IP_ADDRESS = "IP_ADDRESS"
    API_KEY = "API_KEY"
    CREDENTIAL = "CREDENTIAL"
    SECRET = "SECRET"
    PASSWORD = "PASSWORD"
    PRIVATE_KEY = "PRIVATE_KEY"
    JWT = "JWT"
    NAME = "NAME"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    PATH = "PATH"
    CUSTOM = "CUSTOM"
    ALIAS = "ALIAS"
    ENTITY = "ENTITY"
    SYNTHETIC_SIGNAL = "SYNTHETIC_SIGNAL"
    SENSITIVE_PHRASE = "SENSITIVE_PHRASE"


# Standard deterministic replacement placeholder tokens
DEFAULT_REDACTION_TOKENS: dict[PIICategory, str] = {
    PIICategory.EMAIL: "[REDACTED_EMAIL]",
    PIICategory.PHONE: "[REDACTED_PHONE]",
    PIICategory.IP_ADDRESS: "[REDACTED_IP_ADDRESS]",
    PIICategory.API_KEY: "[REDACTED_API_KEY]",
    PIICategory.CREDENTIAL: "[REDACTED_CREDENTIAL]",
    PIICategory.SECRET: "[REDACTED_SECRET]",
    PIICategory.PASSWORD: "[REDACTED_PASSWORD]",
    PIICategory.PRIVATE_KEY: "[REDACTED_PRIVATE_KEY]",
    PIICategory.JWT: "[REDACTED_JWT]",
    PIICategory.NAME: "[REDACTED_NAME]",
    PIICategory.SSN: "[REDACTED_SSN]",
    PIICategory.CREDIT_CARD: "[REDACTED_CREDIT_CARD]",
    PIICategory.PATH: "[REDACTED_PATH]",
    PIICategory.CUSTOM: "[REDACTED_CUSTOM]",
    PIICategory.ALIAS: "[REDACTED_ALIAS]",
    PIICategory.ENTITY: "[REDACTED_ENTITY]",
    PIICategory.SYNTHETIC_SIGNAL: "[SYNTHETIC_SIGNAL]",
    PIICategory.SENSITIVE_PHRASE: "[SYNTHETIC_SIGNAL]",
}


@dataclass(frozen=True)
class PIIFinding:
    """A detected instance of PII or sensitive data."""
    rule: str
    category: PIICategory
    matched_text: str
    start: int
    end: int
    replacement: str
    confidence: float = 1.0
    path: str = "$"

    def to_dict(self) -> dict[str, Any]:
        """Convert finding to serializable dictionary."""
        return {
            "rule": self.rule,
            "category": self.category.value,
            "matched_preview": self.matched_text[:8] + "…" if len(self.matched_text) > 8 else self.matched_text,
            "span": f"{self.start}:{self.end}",
            "start": self.start,
            "end": self.end,
            "replacement": self.replacement,
            "confidence": self.confidence,
            "path": self.path,
        }


@dataclass
class SanitizationReceipt:
    """Auditable cryptographic receipt for a sanitization run."""
    receipt_id: str
    timestamp: str
    input_hash: str
    output_hash: str
    findings_count: int
    findings_by_category: dict[str, int]
    findings: list[dict[str, Any]]
    is_clean: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        if not self.receipt_hash:
            data = {
                "receipt_id": self.receipt_id,
                "timestamp": self.timestamp,
                "input_hash": self.input_hash,
                "output_hash": self.output_hash,
                "findings_count": self.findings_count,
                "findings_by_category": self.findings_by_category,
                "is_clean": self.is_clean,
                "metadata": self.metadata,
            }
            self.receipt_hash = sha256_json(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================
# SENSITIVE PHRASE REGISTRY
# ============================================================================

@dataclass
class SensitivePhraseEntry:
    """An entry in the sensitive phrase registry."""
    phrase: str
    pattern: re.Pattern[str]
    replacement: str
    category: PIICategory
    rule_name: str
    case_sensitive: bool = False
    whole_word: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "pattern": self.pattern.pattern,
            "replacement": self.replacement,
            "category": self.category.value,
            "rule_name": self.rule_name,
            "case_sensitive": self.case_sensitive,
            "whole_word": self.whole_word,
        }


class SensitivePhraseRegistry:
    """Registry of operator-supplied sensitive phrases.

    The registry starts empty. Phrases are deployment-specific, so none ship
    with the package; register them per corpus. Replacement tokens:
    - [REDACTED_ALIAS] (aliases, personal nicknames)
    - [REDACTED_ENTITY] (private entity names, project codenames)
    - [SYNTHETIC_SIGNAL] (any other phrase that must not reach a release)
    """

    def __init__(self) -> None:
        self._entries: list[SensitivePhraseEntry] = []

    def register_phrase(
        self,
        phrase: str,
        replacement: str = "[REDACTED_ALIAS]",
        category: PIICategory = PIICategory.ALIAS,
        case_sensitive: bool = False,
        whole_word: bool = True,
        rule_name: str | None = None,
    ) -> None:
        """Register a single phrase with replacement token and category."""
        clean_phrase = phrase.strip()
        if not clean_phrase:
            return

        flags = 0 if case_sensitive else re.IGNORECASE
        escaped = re.escape(clean_phrase)
        # Support flexible whitespace in multi-word phrases (e.g. "Project  Nightjar")
        escaped_whitespace = re.sub(r"\\ ", r"\\s+", escaped)

        if whole_word:
            pattern_str = r"\b" + escaped_whitespace + r"\b"
        else:
            pattern_str = escaped_whitespace

        pattern = re.compile(pattern_str, flags)
        r_name = rule_name or f"sensitive_{category.value.lower()}_{re.sub(r'[^a-zA-Z0-9_]', '_', clean_phrase.lower())}"

        # Remove existing entry if duplicate phrase registered
        self._entries = [e for e in self._entries if e.phrase.lower() != clean_phrase.lower()]

        entry = SensitivePhraseEntry(
            phrase=clean_phrase,
            pattern=pattern,
            replacement=replacement,
            category=category,
            rule_name=r_name,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
        )
        self._entries.append(entry)

    def register_alias(self, alias: str, replacement: str = "[REDACTED_ALIAS]") -> None:
        """Convenience method to register an alias / personal nickname."""
        self.register_phrase(alias, replacement=replacement, category=PIICategory.ALIAS)

    def register_entity(self, entity: str, replacement: str = "[REDACTED_ENTITY]") -> None:
        """Convenience method to register a sensitive entity codename."""
        self.register_phrase(entity, replacement=replacement, category=PIICategory.ENTITY)

    def register_signal_phrase(self, phrase: str, replacement: str = "[SYNTHETIC_SIGNAL]") -> None:
        """Convenience method to register a phrase redacted to the synthetic-signal token."""
        self.register_phrase(phrase, replacement=replacement, category=PIICategory.SYNTHETIC_SIGNAL)

    def register_regex(
        self,
        pattern: str | re.Pattern[str],
        replacement: str = "[SYNTHETIC_SIGNAL]",
        category: PIICategory = PIICategory.CUSTOM,
        rule_name: str | None = None,
    ) -> None:
        """Register an explicit regular expression pattern."""
        compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
        r_name = rule_name or f"custom_regex_{len(self._entries)}"
        entry = SensitivePhraseEntry(
            phrase=compiled.pattern,
            pattern=compiled,
            replacement=replacement,
            category=category,
            rule_name=r_name,
            case_sensitive=True,
            whole_word=False,
        )
        self._entries.append(entry)

    def register_batch(
        self,
        phrases: Sequence[str] | dict[str, str],
        default_replacement: str = "[REDACTED_ALIAS]",
        category: PIICategory = PIICategory.ALIAS,
    ) -> None:
        """Register multiple phrases in batch."""
        if isinstance(phrases, dict):
            for phrase, repl in phrases.items():
                self.register_phrase(phrase, replacement=repl, category=category)
        else:
            for phrase in phrases:
                self.register_phrase(phrase, replacement=default_replacement, category=category)

    def remove_phrase(self, phrase: str) -> bool:
        """Remove a registered phrase. Returns True if removed."""
        initial_len = len(self._entries)
        self._entries = [e for e in self._entries if e.phrase.lower() != phrase.strip().lower()]
        return len(self._entries) < initial_len

    def clear(self) -> None:
        """Clear all registered sensitive phrases."""
        self._entries.clear()

    def list_phrases(self) -> list[dict[str, Any]]:
        """List all registered phrases as serializable dictionaries."""
        return [e.to_dict() for e in self._entries]

    def scan(self, text: str, path: str = "$") -> list[PIIFinding]:
        """Scan text and return findings for all matching sensitive phrases."""
        if not text or not isinstance(text, str):
            return []

        findings: list[PIIFinding] = []
        for entry in self._entries:
            for match in entry.pattern.finditer(text):
                findings.append(
                    PIIFinding(
                        rule=entry.rule_name,
                        category=entry.category,
                        matched_text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        replacement=entry.replacement,
                        confidence=1.0,
                        path=path,
                    )
                )
        return findings

    def redact(self, text: str, path: str = "$") -> tuple[str, list[PIIFinding]]:
        """Redact registered sensitive phrases in text."""
        findings = self.scan(text, path=path)
        if not findings:
            return text, []

        findings.sort(key=lambda f: (f.start, -(f.end - f.start)))
        non_overlapping: list[PIIFinding] = []
        last_end = -1
        for f in findings:
            if f.start >= last_end:
                non_overlapping.append(f)
                last_end = f.end

        chars = list(text)
        for f in reversed(non_overlapping):
            chars[f.start:f.end] = list(f.replacement)

        return "".join(chars), non_overlapping

    def get_rules(self) -> list[tuple[str, PIICategory, re.Pattern[str], str]]:
        """Export registry entries as rule tuples (rule_name, category, pattern, replacement)."""
        return [(e.rule_name, e.category, e.pattern, e.replacement) for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, phrase: str) -> bool:
        return any(e.phrase.lower() == phrase.strip().lower() for e in self._entries)


# ============================================================================
# PII SANITIZER IMPLEMENTATION
# ============================================================================

class PIISanitizer:
    """Automated, deterministic PII and secret sanitizer."""

    # Built-in compiled patterns
    # IPv4 regex matching valid octets (0-255)
    _IPV4_PATTERN = (
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\b"
    )
    # IPv6 standard, compressed, and mapped addresses
    _IPV6_PATTERN = (
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|"
        r"\b[0-9a-fA-F]{1,4}::[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5}\b|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5}\b|"
        r"::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}\b|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b|"
        r"\b[0-9a-fA-F]{1,4}::\b|"
        r"\b::1\b|"
        r"\b::\b"
    )

    def __init__(
        self,
        token_overrides: dict[PIICategory, str] | None = None,
        custom_names: Sequence[str] | None = None,
        extra_rules: Sequence[tuple[str, PIICategory, re.Pattern[str], str]] | None = None,
        sensitive_registry: SensitivePhraseRegistry | None = None,
        workspace_home: str | None = None,
    ) -> None:
        self.tokens: dict[PIICategory, str] = {**DEFAULT_REDACTION_TOKENS}
        if token_overrides:
            self.tokens.update(token_overrides)

        self.custom_names = set(custom_names or [])
        self.sensitive_registry = (
            sensitive_registry if sensitive_registry is not None else SensitivePhraseRegistry()
        )
        # The operator's *own* home is rewritten to /workspace, preserving path
        # structure, because it is the capturing operator's layout rather than a
        # third party's identity. Every other user's home is opaque (rule 9).
        #
        # This was hardcoded to one operator's directory, so the structure-
        # preserving rewrite only ever worked for the machine it was written on.
        # Any other operator's paths fell through to [REDACTED_PATH] -- private,
        # but not what the rule was for. Defaults to the running user's home so
        # existing behaviour is unchanged where it was already correct; pass an
        # explicit value when sanitising data captured on another machine.
        self.workspace_home = (workspace_home or str(Path.home())).rstrip("/")
        _home = re.escape(self.workspace_home)
        self._workspace_home_pattern = re.compile(
            rf"{_home}(?:\.\.\.|(?:/[a-zA-Z0-9._~-]+)*)?(?![a-zA-Z0-9_-])"
        )
        self._rules: list[tuple[str, PIICategory, re.Pattern[str], str]] = []
        self._setup_rules()

        if extra_rules:
            self._rules.extend(extra_rules)

    def _setup_rules(self) -> None:
        """Register built-in regex rules in prioritized order."""
        # 1. Private Key Blocks (multiline, highest precedence)
        self._rules.append((
            "private_key_block",
            PIICategory.PRIVATE_KEY,
            re.compile(
                r"-----BEGIN (?:[A-Z0-9_-]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z0-9_-]+ )?PRIVATE KEY-----"
            ),
            self.tokens[PIICategory.PRIVATE_KEY],
        ))

        # 2. Known Vendor API Keys & Secret Tokens
        self._rules.append((
            "openai_api_key",
            PIICategory.API_KEY,
            re.compile(r"\bsk-(?:proj-|ant-|live-|svcacct-)?[A-Za-z0-9_-]{20,}\b"),
            self.tokens[PIICategory.API_KEY],
        ))
        self._rules.append((
            "anthropic_api_key",
            PIICategory.API_KEY,
            re.compile(r"\bsk-ant-(?:api\d{2}-)?[A-Za-z0-9_-]{20,}\b"),
            self.tokens[PIICategory.API_KEY],
        ))
        self._rules.append((
            "aws_access_key",
            PIICategory.API_KEY,
            re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"),
            self.tokens[PIICategory.API_KEY],
        ))
        self._rules.append((
            "github_token",
            PIICategory.API_KEY,
            re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{36,}\b"),
            self.tokens[PIICategory.API_KEY],
        ))
        self._rules.append((
            "slack_token",
            PIICategory.API_KEY,
            re.compile(r"\bxox[baprs]-[0-9A-Za-z]{10,}-[0-9A-Za-z]{10,}-[0-9A-Za-z]{20,}\b"),
            self.tokens[PIICategory.API_KEY],
        ))
        self._rules.append((
            "jwt_token",
            PIICategory.JWT,
            re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
            self.tokens[PIICategory.JWT],
        ))

        # 3. Generic Key / Token / Credential / Password assignments
        self._rules.append((
            "generic_api_key_assignment",
            PIICategory.API_KEY,
            re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*['\"]?([A-Za-z0-9+/=_\-.@$!%*?&#]{16,})['\"]?"),
            self.tokens[PIICategory.API_KEY],
        ))
        self._rules.append((
            "generic_secret_assignment",
            PIICategory.SECRET,
            re.compile(r"(?i)\b(?:secret[_-]?key|client[_-]?secret|app[_-]?secret)\s*[:=]\s*['\"]?([A-Za-z0-9+/=_\-.@$!%*?&#]{16,})['\"]?"),
            self.tokens[PIICategory.SECRET],
        ))
        self._rules.append((
            "generic_password_assignment",
            PIICategory.PASSWORD,
            re.compile(r"(?i)\b(?:password|passwd|db[_-]?pass)\s*[:=]\s*['\"]?([A-Za-z0-9+/=_\-.@$!%*?&#]{6,})['\"]?"),
            self.tokens[PIICategory.PASSWORD],
        ))
        self._rules.append((
            "bearer_token",
            PIICategory.CREDENTIAL,
            re.compile(r"(?i)\bBearer\s+([A-Za-z0-9._~+/-]{20,})\b"),
            self.tokens[PIICategory.CREDENTIAL],
        ))

        # 4. Standard Emails
        self._rules.append((
            "email",
            PIICategory.EMAIL,
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            self.tokens[PIICategory.EMAIL],
        ))

        # 5. Phone Numbers (International & Standard formats)
        self._rules.append((
            "phone_international",
            PIICategory.PHONE,
            re.compile(r"\+\d{1,3}[-.\s]?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b"),
            self.tokens[PIICategory.PHONE],
        ))
        self._rules.append((
            "phone_standard",
            PIICategory.PHONE,
            re.compile(r"(?:\(\d{3}\)|\b\d{3})[-.\s]\d{3}[-.\s]\d{4}\b"),
            self.tokens[PIICategory.PHONE],
        ))

        # 6. IP Addresses (IPv4 & IPv6)
        self._rules.append((
            "ipv4_address",
            PIICategory.IP_ADDRESS,
            re.compile(self._IPV4_PATTERN),
            self.tokens[PIICategory.IP_ADDRESS],
        ))
        self._rules.append((
            "ipv6_address",
            PIICategory.IP_ADDRESS,
            re.compile(self._IPV6_PATTERN),
            self.tokens[PIICategory.IP_ADDRESS],
        ))

        # 7. SSN (Social Security Numbers)
        self._rules.append((
            "ssn",
            PIICategory.SSN,
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            self.tokens[PIICategory.SSN],
        ))

        # 8. Credit Card Numbers
        self._rules.append((
            "credit_card",
            PIICategory.CREDIT_CARD,
            re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b"),
            self.tokens[PIICategory.CREDIT_CARD],
        ))

        # 9. Home & User File Paths. The operator's own home is excluded here
        # because workspace_home_path (above) rewrites it to /workspace instead.
        # The lookahead is built from the configured home rather than a literal
        # username, so this holds for whoever is running the sanitiser.
        self._rules.append((
            "user_home_path",
            PIICategory.PATH,
            re.compile(
                rf"(?!{re.escape(self.workspace_home)}\b)"
                r"(?:/home|/Users)/[a-zA-Z0-9._-]+(?:/[a-zA-Z0-9._~-]+)*"
            ),
            self.tokens[PIICategory.PATH],
        ))

        # 10. Person Names (Honorifics & Key-Value Patterns)
        self._rules.append((
            "name_with_honorific",
            PIICategory.NAME,
            re.compile(r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+\b"),
            self.tokens[PIICategory.NAME],
        ))
        self._rules.append((
            "name_labeled_field",
            PIICategory.NAME,
            re.compile(r"(?i)\b(?:full[_-]?name|author|contributor|user[_-]?name|developer|contact|legal[_-]?name|signed-off-by)\s*[:=]\s*['\"]?([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+)['\"]?"),
            self.tokens[PIICategory.NAME],
        ))

    def register_custom_name(self, name: str) -> None:
        """Register an individual person name for redaction."""
        if name.strip():
            self.custom_names.add(name.strip())

    def register_sensitive_phrase(
        self,
        phrase: str,
        replacement: str = "[REDACTED_ALIAS]",
        category: PIICategory = PIICategory.ALIAS,
    ) -> None:
        """Register a sensitive phrase into the active sensitive phrase registry."""
        if self.sensitive_registry is None:
            self.sensitive_registry = SensitivePhraseRegistry()
        self.sensitive_registry.register_phrase(phrase, replacement=replacement, category=category)

    def scan_text(self, text: str, path: str = "$") -> list[PIIFinding]:
        """Scan text and return all detected PII findings."""
        if not text or not isinstance(text, str):
            return []

        findings: list[PIIFinding] = []

        # 1. Custom registered names (legacy support)
        for name in self.custom_names:
            pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
            for m in pattern.finditer(text):
                findings.append(
                    PIIFinding(
                        rule="custom_name",
                        category=PIICategory.NAME,
                        matched_text=m.group(0),
                        start=m.start(),
                        end=m.end(),
                        replacement=self.tokens[PIICategory.NAME],
                        confidence=1.0,
                        path=path,
                    )
                )

        # 2. Sensitive phrase registry (operator-registered aliases, entities, phrases)
        if self.sensitive_registry:
            findings.extend(self.sensitive_registry.scan(text, path=path))

        # 3. Path conversion: the operator's own home -> /workspace/...
        for match in self._workspace_home_pattern.finditer(text):
            matched_str = match.group(0)
            remainder = matched_str[len(self.workspace_home):]
            replacement = f"/workspace{remainder}"
            findings.append(
                PIIFinding(
                    rule="workspace_home_path",
                    category=PIICategory.PATH,
                    matched_text=matched_str,
                    start=match.start(),
                    end=match.end(),
                    replacement=replacement,
                    confidence=1.0,
                    path=path,
                )
            )

        # 4. Built-in and extra rules
        for rule_name, category, pattern, replacement in self._rules:
            for match in pattern.finditer(text):
                # If the rule has capture groups (like assignment values), target the captured secret value
                if match.groups() and match.lastindex and match.group(1):
                    start, end = match.span(1)
                    matched_str = match.group(1)
                else:
                    start, end = match.span(0)
                    matched_str = match.group(0)

                findings.append(
                    PIIFinding(
                        rule=rule_name,
                        category=category,
                        matched_text=matched_str,
                        start=start,
                        end=end,
                        replacement=replacement,
                        confidence=0.95,
                        path=path,
                    )
                )

        # Sort findings by start position ascending, and longer matches first
        findings.sort(key=lambda f: (f.start, -(f.end - f.start)))
        return findings

    def redact_text(self, text: str, path: str = "$") -> tuple[str, list[PIIFinding]]:
        """Redact all PII in text using deterministic replacement tokens.

        Returns (sanitized_text, list_of_applied_findings).
        """
        if not text or not isinstance(text, str):
            return text, []

        findings = self.scan_text(text, path=path)
        if not findings:
            return text, []

        # Deduplicate overlapping spans: select highest priority non-overlapping intervals
        non_overlapping: list[PIIFinding] = []
        last_end = -1
        for f in findings:
            if f.start >= last_end:
                non_overlapping.append(f)
                last_end = f.end

        # Reconstruct text from right to left to maintain offset stability
        chars = list(text)
        for f in reversed(non_overlapping):
            chars[f.start:f.end] = list(f.replacement)

        redacted_text = "".join(chars)
        return redacted_text, non_overlapping

    def scan_object(self, obj: Any, path: str = "$") -> list[PIIFinding]:
        """Recursively scan an arbitrary Python object / JSON tree for PII."""
        findings: list[PIIFinding] = []
        if isinstance(obj, str):
            findings.extend(self.scan_text(obj, path=path))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                current_path = f"{path}.{k}"
                if isinstance(k, str):
                    findings.extend(self.scan_text(k, path=f"{path}.key({k})"))
                findings.extend(self.scan_object(v, path=current_path))
        elif isinstance(obj, (list, tuple, set)):
            for idx, item in enumerate(obj):
                current_path = f"{path}[{idx}]"
                findings.extend(self.scan_object(item, path=current_path))
        return findings

    def redact_object(self, obj: Any, path: str = "$") -> tuple[Any, list[PIIFinding]]:
        """Recursively redact all strings in an arbitrary Python object / JSON tree."""
        all_findings: list[PIIFinding] = []

        def _walk(item: Any, curr_path: str) -> Any:
            if isinstance(item, str):
                cleaned, findings = self.redact_text(item, path=curr_path)
                all_findings.extend(findings)
                return cleaned
            elif isinstance(item, dict):
                cleaned_dict: dict[Any, Any] = {}
                for k, v in item.items():
                    k_cleaned = _walk(k, f"{curr_path}.key({k})") if isinstance(k, str) else k
                    cleaned_dict[k_cleaned] = _walk(v, f"{curr_path}.{k}")
                return cleaned_dict
            elif isinstance(item, list):
                return [_walk(elem, f"{curr_path}[{i}]") for i, elem in enumerate(item)]
            elif isinstance(item, tuple):
                return tuple(_walk(elem, f"{curr_path}[{i}]") for i, elem in enumerate(item))
            elif isinstance(item, set):
                return {_walk(elem, f"{curr_path}[*]") for elem in item}
            else:
                return item

        sanitized = _walk(obj, path)
        return sanitized, all_findings

    def sanitize_trace(
        self,
        trace: dict[str, Any],
        trace_id_key: str = "trace_id",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], SanitizationReceipt]:
        """Sanitize a trace record and generate an auditable cryptographic receipt."""
        raw_hash = sha256_json(trace)
        sanitized_trace, findings = self.redact_object(trace)
        out_hash = sha256_json(sanitized_trace)

        # Count findings by category
        cat_counts: dict[str, int] = {}
        for f in findings:
            cat_counts[f.category.value] = cat_counts.get(f.category.value, 0) + 1

        tid = trace.get(trace_id_key) or trace.get("id") or f"trace-{raw_hash[:8]}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Double check that no residual PII remains
        rescan_findings = self.scan_object(sanitized_trace)
        is_clean = len(rescan_findings) == 0

        receipt = SanitizationReceipt(
            receipt_id=f"receipt-{tid}",
            timestamp=now,
            input_hash=raw_hash,
            output_hash=out_hash,
            findings_count=len(findings),
            findings_by_category=cat_counts,
            findings=[f.to_dict() for f in findings],
            is_clean=is_clean,
            metadata=metadata or {},
        )
        return sanitized_trace, receipt


# ============================================================================
# LEGAL LICENSE COMPLIANCE VERIFICATION
# ============================================================================

class LicenseCategory(str, Enum):
    """Classification category of software licenses."""
    PERMISSIVE = "PERMISSIVE"       # MIT, Apache 2.0, BSD, Unlicense, CC0, ISC, etc.
    COPYLEFT = "COPYLEFT"           # GPL, AGPL, LGPL, SSPL, EUPL, MPL, etc.
    PROPRIETARY = "PROPRIETARY"     # Commercial, All Rights Reserved, Closed, Confidential
    UNKNOWN = "UNKNOWN"             # Missing, Unrecognized or Ambiguous


@dataclass
class LicenseVerificationResult:
    """Auditable result of verifying a software license."""
    is_compliant: bool
    license_id: str | None
    license_name: str
    category: LicenseCategory
    reason: str
    confidence: float
    spdx_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_compliant": self.is_compliant,
            "license_id": self.license_id,
            "license_name": self.license_name,
            "category": self.category.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "spdx_id": self.spdx_id,
            "details": self.details,
        }


@dataclass
class RepositoryLicenseReport:
    """Auditable compliance report for an entire repository or source pack."""
    repo_path: str
    is_compliant: bool
    detected_licenses: list[LicenseVerificationResult]
    rejected_reasons: list[str]
    files_inspected: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    report_hash: str = ""

    def __post_init__(self) -> None:
        if not self.report_hash:
            data = {
                "repo_path": self.repo_path,
                "is_compliant": self.is_compliant,
                "detected_licenses": [l.to_dict() for l in self.detected_licenses],
                "rejected_reasons": self.rejected_reasons,
                "files_inspected": self.files_inspected,
                "timestamp": self.timestamp,
            }
            self.report_hash = sha256_json(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "is_compliant": self.is_compliant,
            "detected_licenses": [l.to_dict() for l in self.detected_licenses],
            "rejected_reasons": self.rejected_reasons,
            "files_inspected": self.files_inspected,
            "timestamp": self.timestamp,
            "report_hash": self.report_hash,
        }


class LicenseVerifier:
    """Strict legal license compliance verifier.

    Enforces that all source code, models, and repositories carry permissive
    open-source licenses (MIT, Apache-2.0, BSD-2/3-Clause, Unlicense, CC0, ISC, 0BSD)
    and actively rejects GPL copyleft (GPL, AGPL, LGPL), SSPL, MPL, and proprietary/commercial terms.
    """

    # Permissive standard licenses (Allowed)
    PERMISSIVE_SPDX: set[str] = {
        "MIT",
        "MIT-0",
        "MIT-MODERN-VARIANT",
        "APACHE-2.0",
        "APACHE-1.1",
        "APACHE-1.0",
        "BSD-2-CLAUSE",
        "BSD-3-CLAUSE",
        "BSD-3-CLAUSE-CLEAR",
        "BSD-4-CLAUSE",
        "0BSD",
        "UNLICENSE",
        "CC0-1.0",
        "ISC",
        "PYTHON-2.0",
        "PSF-2.0",
        "BSL-1.0",
        "ZLIB",
        "WTFPL",
        "ARTISTIC-2.0",
    }

    # Copyleft / Restrictive licenses (Rejected)
    COPYLEFT_SPDX: set[str] = {
        "GPL-1.0", "GPL-1.0-ONLY", "GPL-1.0-OR-LATER",
        "GPL-2.0", "GPL-2.0-ONLY", "GPL-2.0-OR-LATER",
        "GPL-3.0", "GPL-3.0-ONLY", "GPL-3.0-OR-LATER",
        "AGPL-1.0", "AGPL-1.0-ONLY", "AGPL-1.0-OR-LATER",
        "AGPL-3.0", "AGPL-3.0-ONLY", "AGPL-3.0-OR-LATER",
        "LGPL-2.0", "LGPL-2.0-ONLY", "LGPL-2.0-OR-LATER",
        "LGPL-2.1", "LGPL-2.1-ONLY", "LGPL-2.1-OR-LATER",
        "LGPL-3.0", "LGPL-3.0-ONLY", "LGPL-3.0-OR-LATER",
        "SSPL-1.0", "SSPL",
        "EUPL-1.1", "EUPL-1.2",
        "MPL-1.0", "MPL-1.1", "MPL-2.0",
        "CDDL-1.0", "CDDL-1.1",
        "EPL-1.0", "EPL-2.0",
        "OSL-1.0", "OSL-2.0", "OSL-3.0",
    }

    # Proprietary terms & phrases (Rejected)
    PROPRIETARY_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"(?i)\ball\s+rights\s+reserved\b"),
        re.compile(r"(?i)\bproprietary\b"),
        re.compile(r"(?i)\bconfidential\b"),
        re.compile(r"(?i)\bcommercial\s+license\b"),
        re.compile(r"(?i)\bunauthorized\s+(?:copying|distribution|reproduction|use)\b"),
        re.compile(r"(?i)\b(?:strictly\s+)?prohibited\b"),
        re.compile(r"(?i)\bnot\s+for\s+commercial\s+use\b"),
        re.compile(r"(?i)\bnon-commercial\s+use\s+only\b"),
        re.compile(r"(?i)\bno\s+license\s+granted\b"),
    ]

    # Text signatures for heuristic license detection
    LICENSE_SIGNATURES: list[tuple[str, str, LicenseCategory, Sequence[re.Pattern[str]]]] = [
        (
            "MIT",
            "MIT License",
            LicenseCategory.PERMISSIVE,
            [
                re.compile(r"(?i)permission\s+is\s+hereby\s+granted,\s+free\s+of\s+charge,\s+to\s+any\s+person\s+obtaining\s+a\s+copy"),
                re.compile(r"(?i)the\s+above\s+copyright\s+notice\s+and\s+this\s+permission\s+notice\s+shall\s+be\s+included"),
            ],
        ),
        (
            "Apache-2.0",
            "Apache License 2.0",
            LicenseCategory.PERMISSIVE,
            [
                re.compile(r"(?i)apache\s+license,\s+version\s+2\.0"),
                re.compile(r"(?i)http://www\.apache\.org/licenses/LICENSE-2\.0"),
                re.compile(r"(?i)licensed\s+under\s+the\s+apache\s+license,\s+version\s+2\.0"),
            ],
        ),
        (
            "BSD-3-Clause",
            "BSD 3-Clause 'New' or 'Revised' License",
            LicenseCategory.PERMISSIVE,
            [
                re.compile(r"(?i)redistribution\s+and\s+use\s+in\s+source\s+and\s+binary\s+forms"),
                re.compile(r"(?i)neither\s+the\s+name\s+of\s+(?:the\s+copyright\s+holder|.*)\s+nor\s+the\s+names\s+of\s+its\s+contributors"),
            ],
        ),
        (
            "BSD-2-Clause",
            "BSD 2-Clause 'Simplified' License",
            LicenseCategory.PERMISSIVE,
            [
                re.compile(r"(?i)redistribution\s+and\s+use\s+in\s+source\s+and\s+binary\s+forms"),
                re.compile(r"(?i)redistributions\s+in\s+binary\s+form\s+must\s+reproduce\s+the\s+above\s+copyright\s+notice"),
            ],
        ),
        (
            "Unlicense",
            "The Unlicense (Public Domain)",
            LicenseCategory.PERMISSIVE,
            [
                re.compile(r"(?i)this\s+is\s+free\s+and\s+unencumbered\s+software\s+released\s+into\s+the\s+public\s+domain"),
                re.compile(r"(?i)anyone\s+is\s+free\s+to\s+copy,\s+modify,\s+publish,\s+use,\s+compile,\s+sell,\s+or\s+distribute"),
            ],
        ),
        (
            "CC0-1.0",
            "Creative Commons Zero v1.0 Universal",
            LicenseCategory.PERMISSIVE,
            [
                re.compile(r"(?i)creative\s+commons\s+zero\s+v1\.0\s+universal"),
                re.compile(r"(?i)public\s+domain\s+dedication"),
                re.compile(r"(?i)creativecommons\.org/publicdomain/zero/1\.0"),
            ],
        ),
        (
            "0BSD",
            "BSD Zero Clause License",
            LicenseCategory.PERMISSIVE,
            [
                re.compile(r"(?i)zero\s+clause\s+bsd"),
                re.compile(r"(?i)permission\s+to\s+use,\s+copy,\s+modify,\s+and/or\s+distribute\s+this\s+software\s+for\s+any\s+purpose\s+with\s+or\s+without\s+fee"),
            ],
        ),
        (
            "GPL-3.0",
            "GNU General Public License v3.0",
            LicenseCategory.COPYLEFT,
            [
                re.compile(r"(?i)gnu\s+general\s+public\s+license"),
                re.compile(r"(?i)version\s+3,\s+29\s+june\s+2007"),
                re.compile(r"(?i)free\s+software\s+foundation,\s+inc\."),
            ],
        ),
        (
            "GPL-2.0",
            "GNU General Public License v2.0",
            LicenseCategory.COPYLEFT,
            [
                re.compile(r"(?i)gnu\s+general\s+public\s+license"),
                re.compile(r"(?i)version\s+2,\s+june\s+1991"),
            ],
        ),
        (
            "AGPL-3.0",
            "GNU Affero General Public License v3.0",
            LicenseCategory.COPYLEFT,
            [
                re.compile(r"(?i)gnu\s+affero\s+general\s+public\s+license"),
            ],
        ),
        (
            "LGPL-3.0",
            "GNU Lesser General Public License v3.0",
            LicenseCategory.COPYLEFT,
            [
                re.compile(r"(?i)gnu\s+lesser\s+general\s+public\s+license"),
            ],
        ),
    ]

    def __init__(self, allowed_licenses: set[str] | None = None) -> None:
        self.allowed_spdx = set(allowed_licenses) if allowed_licenses else self.PERMISSIVE_SPDX

    def normalize_identifier(self, identifier: str) -> str:
        """Normalize a license name or SPDX tag for consistent lookup."""
        if not identifier or not isinstance(identifier, str):
            return ""
        norm = identifier.strip().upper()
        # Common aliases & normalization
        aliases = {
            "APACHE 2.0": "APACHE-2.0",
            "APACHE 2": "APACHE-2.0",
            "APACHE2": "APACHE-2.0",
            "APACHE LICENSE 2.0": "APACHE-2.0",
            "APACHE LICENSE, VERSION 2.0": "APACHE-2.0",
            "ASL 2.0": "APACHE-2.0",
            "BSD 3-CLAUSE": "BSD-3-CLAUSE",
            "BSD-3": "BSD-3-CLAUSE",
            "BSD 2-CLAUSE": "BSD-2-CLAUSE",
            "BSD-2": "BSD-2-CLAUSE",
            "THE UNLICENSE": "UNLICENSE",
            "PUBLIC DOMAIN": "UNLICENSE",
            "CC0": "CC0-1.0",
            "GPLV3": "GPL-3.0",
            "GPLV2": "GPL-2.0",
            "AGPLV3": "AGPL-3.0",
            "LGPLV3": "LGPL-3.0",
            "LGPLV2.1": "LGPL-2.1",
        }
        return aliases.get(norm, norm.replace(" ", "-").replace("_", "-"))

    def verify_license_identifier(self, license_str: str) -> LicenseVerificationResult:
        """Verify a license by its SPDX tag or common identifier."""
        if not license_str or not license_str.strip():
            return LicenseVerificationResult(
                is_compliant=False,
                license_id=None,
                license_name="Unspecified",
                category=LicenseCategory.UNKNOWN,
                reason="License identifier is empty or missing (fail-closed compliance policy).",
                confidence=1.0,
            )

        raw = license_str.strip()
        norm = self.normalize_identifier(raw)

        # 1. Check if explicitly in permissive set
        if norm in self.allowed_spdx:
            return LicenseVerificationResult(
                is_compliant=True,
                license_id=norm,
                license_name=raw,
                category=LicenseCategory.PERMISSIVE,
                reason=f"Permissive license '{norm}' is approved for commercial use and redistribution.",
                confidence=1.0,
                spdx_id=norm,
            )

        # 2. Check if explicitly in copyleft set
        if norm in self.COPYLEFT_SPDX or any(norm.startswith(cp) for cp in ("GPL", "AGPL", "LGPL", "SSPL", "EUPL", "MPL")):
            return LicenseVerificationResult(
                is_compliant=False,
                license_id=norm,
                license_name=raw,
                category=LicenseCategory.COPYLEFT,
                reason=f"Rejected copyleft license '{norm}'. GPL/AGPL/LGPL/SSPL copyleft terms contaminate derivative works.",
                confidence=1.0,
                spdx_id=norm,
            )

        # 3. Check for proprietary indicators
        for pattern in self.PROPRIETARY_PATTERNS:
            if pattern.search(raw):
                return LicenseVerificationResult(
                    is_compliant=False,
                    license_id="PROPRIETARY",
                    license_name=raw,
                    category=LicenseCategory.PROPRIETARY,
                    reason=f"Rejected proprietary source: '{raw}' carries restrictive or non-permissive terms.",
                    confidence=1.0,
                )

        # 4. Unknown / unverified license
        return LicenseVerificationResult(
            is_compliant=False,
            license_id=norm,
            license_name=raw,
            category=LicenseCategory.UNKNOWN,
            reason=f"Unknown or unverified license '{raw}'. Fail-closed: must explicitly be permissive (MIT, Apache 2.0, BSD, Unlicense).",
            confidence=0.5,
        )

    def extract_spdx_header(self, content: str) -> str | None:
        """Extract SPDX-License-Identifier from source comments."""
        match = re.search(r"SPDX-License-Identifier:\s*([A-Za-z0-9._\-+]+)", content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def verify_license_text(self, text: str) -> LicenseVerificationResult:
        """Verify full license text content or file headers."""
        if not text or not text.strip():
            return LicenseVerificationResult(
                is_compliant=False,
                license_id=None,
                license_name="Empty",
                category=LicenseCategory.UNKNOWN,
                reason="License text is empty.",
                confidence=1.0,
            )

        # 1. Check for SPDX tag in text first
        spdx = self.extract_spdx_header(text)
        if spdx:
            return self.verify_license_identifier(spdx)

        # 2. Check for proprietary text signatures
        for pattern in self.PROPRIETARY_PATTERNS:
            if pattern.search(text):
                return LicenseVerificationResult(
                    is_compliant=False,
                    license_id="PROPRIETARY",
                    license_name="Proprietary Terms",
                    category=LicenseCategory.PROPRIETARY,
                    reason="Text contains proprietary or commercial restrictions.",
                    confidence=0.9,
                )

        # 3. Match against known license signatures
        best_match: tuple[str, str, LicenseCategory, int] | None = None
        for spdx_id, name, cat, patterns in self.LICENSE_SIGNATURES:
            matches_count = sum(1 for p in patterns if p.search(text))
            if matches_count > 0:
                if best_match is None or matches_count > best_match[3]:
                    best_match = (spdx_id, name, cat, matches_count)

        if best_match:
            spdx_id, name, cat, count = best_match
            is_compliant = (cat == LicenseCategory.PERMISSIVE)
            reason = (
                f"Detected permissive license '{name}' from text signature."
                if is_compliant
                else f"Detected non-permissive license '{name}' from text signature. Rejected."
            )
            return LicenseVerificationResult(
                is_compliant=is_compliant,
                license_id=spdx_id,
                license_name=name,
                category=cat,
                reason=reason,
                confidence=0.95 if count >= 2 else 0.8,
                spdx_id=spdx_id,
            )

        # Fail-closed
        return LicenseVerificationResult(
            is_compliant=False,
            license_id=None,
            license_name="Unrecognized License Text",
            category=LicenseCategory.UNKNOWN,
            reason="Unrecognized license text. Cannot verify permissive status (fail-closed).",
            confidence=0.4,
        )

    def verify_source_metadata(self, metadata: dict[str, Any]) -> LicenseVerificationResult:
        """Verify license from metadata dictionary (e.g. package.json, pyproject.toml, trace metadata)."""
        # Look for typical license keys
        lic = metadata.get("license") or metadata.get("license_type") or metadata.get("spdx_id")
        if isinstance(lic, str):
            return self.verify_license_identifier(lic)
        elif isinstance(lic, dict):
            lic_type = lic.get("type") or lic.get("name") or lic.get("identifier") or ""
            return self.verify_license_identifier(str(lic_type))
        elif isinstance(lic, list):
            # Check all listed licenses
            results = [self.verify_license_identifier(str(item)) for item in lic]
            if any(not r.is_compliant for r in results):
                bad = [r.license_name for r in results if not r.is_compliant]
                return LicenseVerificationResult(
                    is_compliant=False,
                    license_id=None,
                    license_name="Multiple",
                    category=LicenseCategory.COPYLEFT if any(r.category == LicenseCategory.COPYLEFT for r in results) else LicenseCategory.UNKNOWN,
                    reason=f"Source contains non-compliant license(s): {', '.join(bad)}",
                    confidence=1.0,
                    details={"all_results": [r.to_dict() for r in results]},
                )
            if results:
                return results[0]

        return LicenseVerificationResult(
            is_compliant=False,
            license_id=None,
            license_name="Missing",
            category=LicenseCategory.UNKNOWN,
            reason="No valid license field found in source metadata (fail-closed policy).",
            confidence=1.0,
        )

    def verify_repository(self, repo_path: Path | str) -> RepositoryLicenseReport:
        """Inspect a repository folder for license files, manifest declarations, and headers."""
        root = Path(repo_path)
        if not root.exists() or not root.is_dir():
            return RepositoryLicenseReport(
                repo_path=str(repo_path),
                is_compliant=False,
                detected_licenses=[],
                rejected_reasons=[f"Repository path does not exist or is not a directory: {repo_path}"],
                files_inspected=[],
            )

        detected_results: list[LicenseVerificationResult] = []
        files_inspected: list[str] = []
        rejected_reasons: list[str] = []

        # 1. Search root for standard license files
        license_filenames = {
            "license", "license.txt", "license.md", "licence", "licence.txt",
            "licence.md", "copying", "copying.txt", "unlicense", "notice",
        }

        found_license_files = False
        for p in sorted(root.iterdir()):
            if p.is_file() and p.name.lower() in license_filenames:
                found_license_files = True
                files_inspected.append(str(p.name))
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    res = self.verify_license_text(content)
                    res.details["source_file"] = str(p.name)
                    detected_results.append(res)
                except Exception as e:
                    rejected_reasons.append(f"Failed to read license file {p.name}: {e}")

        # 2. Check standard manifest files
        manifests = [
            ("package.json", self._parse_package_json),
            ("pyproject.toml", self._parse_pyproject_toml),
            ("Cargo.toml", self._parse_cargo_toml),
        ]
        for manifest_name, parser in manifests:
            m_path = root / manifest_name
            if m_path.is_file():
                files_inspected.append(manifest_name)
                try:
                    lic_str = parser(m_path)
                    if lic_str:
                        res = self.verify_license_identifier(lic_str)
                        res.details["source_file"] = manifest_name
                        detected_results.append(res)
                except Exception as e:
                    pass

        # 3. Check sample source files for SPDX headers
        source_exts = {".py", ".ts", ".js", ".go", ".rs", ".cpp", ".c", ".h", ".java", ".sh"}
        checked_source_count = 0
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in source_exts and not any(part.startswith(".") for part in path.parts):
                checked_source_count += 1
                if checked_source_count > 25:
                    break
                try:
                    head = path.read_text(encoding="utf-8", errors="ignore")[:2000]
                    spdx = self.extract_spdx_header(head)
                    if spdx:
                        files_inspected.append(path.relative_to(root).as_posix())
                        res = self.verify_license_identifier(spdx)
                        res.details["source_file"] = path.relative_to(root).as_posix()
                        detected_results.append(res)
                except Exception:
                    pass

        # Evaluate repository compliance
        if not detected_results:
            is_compliant = False
            rejected_reasons.append("No license file, SPDX header, or manifest license declaration found in repository.")
        else:
            # Fail closed: If ANY detected license is COPYLEFT or PROPRIETARY, reject repository
            has_copyleft = any(r.category == LicenseCategory.COPYLEFT for r in detected_results)
            has_proprietary = any(r.category == LicenseCategory.PROPRIETARY for r in detected_results)
            has_permissive = any(r.category == LicenseCategory.PERMISSIVE for r in detected_results)

            if has_copyleft:
                is_compliant = False
                copyleft_names = [r.license_name for r in detected_results if r.category == LicenseCategory.COPYLEFT]
                rejected_reasons.append(f"Repository contains copyleft license(s): {', '.join(copyleft_names)}")
            elif has_proprietary:
                is_compliant = False
                prop_names = [r.license_name for r in detected_results if r.category == LicenseCategory.PROPRIETARY]
                rejected_reasons.append(f"Repository contains proprietary terms: {', '.join(prop_names)}")
            elif not has_permissive:
                is_compliant = False
                rejected_reasons.append("Repository lacks a verifiable permissive open-source license.")
            else:
                is_compliant = True

        return RepositoryLicenseReport(
            repo_path=str(repo_path),
            is_compliant=is_compliant,
            detected_licenses=detected_results,
            rejected_reasons=rejected_reasons,
            files_inspected=files_inspected,
        )

    def _parse_package_json(self, path: Path) -> str | None:
        data = json.loads(path.read_text(encoding="utf-8"))
        lic = data.get("license")
        if isinstance(lic, str):
            return lic
        elif isinstance(lic, dict):
            return lic.get("type")
        return None

    def _parse_pyproject_toml(self, path: Path) -> str | None:
        content = path.read_text(encoding="utf-8")
        # Match license = "MIT" or license = { text = "MIT" }
        m1 = re.search(r"""license\s*=\s*['"]([A-Za-z0-9._\-+ ]+)['"]""", content)
        if m1:
            return m1.group(1)
        m2 = re.search(r"""text\s*=\s*['"]([A-Za-z0-9._\-+ ]+)['"]""", content)
        if m2:
            return m2.group(1)
        return None

    def _parse_cargo_toml(self, path: Path) -> str | None:
        content = path.read_text(encoding="utf-8")
        m = re.search(r"""license\s*=\s*['"]([A-Za-z0-9._\-+ /]+)['"]""", content)
        if m:
            return m.group(1)
        return None


# ============================================================================
# HIGH-LEVEL COMPLIANCE AUDITOR & ENTRYPOINTS
# ============================================================================

class ComplianceAuditor:
    """Unified PII & Legal Compliance Auditor for data refinery operations."""

    def __init__(
        self,
        sanitizer: PIISanitizer | None = None,
        verifier: LicenseVerifier | None = None,
    ) -> None:
        self.sanitizer = sanitizer or PIISanitizer()
        self.verifier = verifier or LicenseVerifier()

    def audit_and_sanitize_record(
        self,
        record: dict[str, Any],
        source_license: str | None = None,
        record_id: str | None = None,
    ) -> dict[str, Any]:
        """Audit record for license compliance and sanitize all PII / secrets."""
        # 1. License Check
        lic_result = None
        if source_license:
            lic_result = self.verifier.verify_license_identifier(source_license)
        elif "license" in record or "license_type" in record:
            lic_result = self.verifier.verify_source_metadata(record)
        else:
            lic_result = LicenseVerificationResult(
                is_compliant=True,
                license_id="INTERNAL",
                license_name="Internal Record",
                category=LicenseCategory.PERMISSIVE,
                reason="No external license requirement specified.",
                confidence=1.0,
            )

        # 2. PII Sanitization
        cleaned_record, receipt = self.sanitizer.sanitize_trace(
            record,
            trace_id_key=record_id or "id",
            metadata={"source_license": source_license, "license_compliant": lic_result.is_compliant},
        )

        # Overall compliance determination
        overall_compliant = lic_result.is_compliant and receipt.is_clean

        audit_payload = {
            "record": cleaned_record,
            "is_compliant": overall_compliant,
            "license_verification": lic_result.to_dict(),
            "sanitization_receipt": receipt.to_dict(),
            "audit_hash": sha256_json({
                "cleaned_hash": receipt.output_hash,
                "license_status": lic_result.is_compliant,
                "receipt_hash": receipt.receipt_hash,
            }),
        }
        return audit_payload


# Global convenience helper functions
_default_sanitizer = PIISanitizer()
_default_verifier = LicenseVerifier()


def scan_pii(text: str) -> list[dict[str, Any]]:
    """Convenience helper to scan text for PII."""
    findings = _default_sanitizer.scan_text(text)
    return [f.to_dict() for f in findings]


def redact_pii(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Convenience helper to redact PII in text."""
    redacted, findings = _default_sanitizer.redact_text(text)
    return redacted, [f.to_dict() for f in findings]


def verify_license(license_id_or_text: str) -> dict[str, Any]:
    """Convenience helper to verify license identifier or text."""
    if len(license_id_or_text.splitlines()) > 1 or len(license_id_or_text) > 80:
        res = _default_verifier.verify_license_text(license_id_or_text)
    else:
        res = _default_verifier.verify_license_identifier(license_id_or_text)
    return res.to_dict()


def verify_repository_license(repo_path: str | Path) -> dict[str, Any]:
    """Convenience helper to audit repository license compliance."""
    report = _default_verifier.verify_repository(repo_path)
    return report.to_dict()


def sanitize_file(
    input_path: str | Path,
    output_path: str | Path,
    receipt_path: str | Path | None = None,
    sanitizer: PIISanitizer | None = None,
) -> SanitizationReceipt:
    """Scrub raw JSON, JSONL, or text file and write cryptographic before/after audit receipts.

    Args:
        input_path: Source file to sanitize (JSON, JSONL, or plain text).
        output_path: Destination path for sanitized output.
        receipt_path: Optional destination path for cryptographic audit receipt.
                      Defaults to '<output_path>.receipt.json'.
        sanitizer: Optional custom PIISanitizer instance.

    Returns:
        SanitizationReceipt with input/output SHA-256 hashes and detailed findings.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)

    if not in_p.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    active_sanitizer = sanitizer or _default_sanitizer

    raw_bytes = in_p.read_bytes()
    in_hash = sha256_bytes(raw_bytes)
    raw_text = raw_bytes.decode("utf-8", errors="replace")

    all_findings: list[PIIFinding] = []
    file_type = "text"

    # Determine file format: check extension or try JSON parsing
    suffix = in_p.suffix.lower()
    is_jsonl = suffix in (".jsonl", ".ndjson")

    if is_jsonl:
        file_type = "jsonl"
        lines = raw_text.splitlines()
        cleaned_lines: list[str] = []
        for idx, line in enumerate(lines):
            line_str = line.strip()
            if not line_str:
                cleaned_lines.append("")
                continue
            try:
                line_obj = json.loads(line_str)
                cleaned_obj, findings = active_sanitizer.redact_object(line_obj, path=f"$[{idx}]")
                all_findings.extend(findings)
                cleaned_lines.append(json.dumps(cleaned_obj, ensure_ascii=False))
            except Exception:
                cleaned_line, findings = active_sanitizer.redact_text(line, path=f"$[{idx}]")
                all_findings.extend(findings)
                cleaned_lines.append(cleaned_line)
        cleaned_text = "\n".join(cleaned_lines)
        if raw_text.endswith("\n"):
            cleaned_text += "\n"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(cleaned_text, encoding="utf-8")

    elif suffix == ".json":
        file_type = "json"
        try:
            parsed = json.loads(raw_text)
            cleaned_obj, findings = active_sanitizer.redact_object(parsed, path="$")
            all_findings.extend(findings)
            cleaned_text = json.dumps(cleaned_obj, indent=2, ensure_ascii=False) + "\n"
        except Exception:
            cleaned_text, findings = active_sanitizer.redact_text(raw_text, path="$")
            all_findings.extend(findings)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(cleaned_text, encoding="utf-8")

    else:
        # Check if content happens to be valid JSON even without .json extension
        is_parsed_json = False
        if raw_text.strip().startswith(("{", "[")):
            try:
                parsed = json.loads(raw_text)
                cleaned_obj, findings = active_sanitizer.redact_object(parsed, path="$")
                all_findings.extend(findings)
                cleaned_text = json.dumps(cleaned_obj, indent=2, ensure_ascii=False) + "\n"
                file_type = "json"
                is_parsed_json = True
            except Exception:
                pass

        if not is_parsed_json:
            file_type = "text"
            cleaned_text, findings = active_sanitizer.redact_text(raw_text, path="$")
            all_findings.extend(findings)

        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(cleaned_text, encoding="utf-8")

    out_bytes = out_p.read_bytes()
    out_hash = sha256_bytes(out_bytes)

    # Compute findings counts
    cat_counts: dict[str, int] = {}
    for f in all_findings:
        cat_counts[f.category.value] = cat_counts.get(f.category.value, 0) + 1

    # Verify no residual PII remains (fail-closed integrity)
    rescan_findings = active_sanitizer.scan_text(out_p.read_text(encoding="utf-8", errors="replace"))
    is_clean = len(rescan_findings) == 0

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt = SanitizationReceipt(
        receipt_id=f"file-receipt-{in_hash[:12]}",
        timestamp=now,
        input_hash=in_hash,
        output_hash=out_hash,
        findings_count=len(all_findings),
        findings_by_category=cat_counts,
        findings=[f.to_dict() for f in all_findings],
        is_clean=is_clean,
        metadata={
            "input_file": str(in_p.resolve()),
            "output_file": str(out_p.resolve()),
            "file_type": file_type,
            "input_size_bytes": len(raw_bytes),
            "output_size_bytes": len(out_bytes),
        },
    )

    # Write receipt to file
    rcpt_path = Path(receipt_path) if receipt_path else Path(f"{output_path}.receipt.json")
    rcpt_path.parent.mkdir(parents=True, exist_ok=True)
    rcpt_path.write_text(json.dumps(receipt.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    return receipt


def main() -> int:
    """CLI entrypoint for batch file sanitization."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="goldtrace-sanitizer",
        description="Gold Trace Mint — Privacy & Sensitive Phrase Sanitizer",
    )
    parser.add_argument("input_path", help="Path to input raw JSON, JSONL, or text file")
    parser.add_argument("output_path", help="Path to output sanitized file")
    parser.add_argument(
        "--receipt-path",
        help="Optional path to output cryptographic receipt JSON (defaults to <output_path>.receipt.json)",
        default=None,
    )
    parser.add_argument(
        "--extra-name",
        action="append",
        default=[],
        help="Additional personal names to redact to [REDACTED_NAME]",
    )
    parser.add_argument(
        "--extra-phrase",
        action="append",
        default=[],
        help="Sensitive phrase to redact (repeatable)",
    )
    parser.add_argument(
        "--phrase-token",
        default="[REDACTED_ALIAS]",
        help="Replacement token for extra phrases (default: [REDACTED_ALIAS])",
    )

    args = parser.parse_args()

    registry = SensitivePhraseRegistry()
    for phrase in args.extra_phrase:
        registry.register_phrase(phrase, replacement=args.phrase_token)

    sanitizer = PIISanitizer(
        custom_names=args.extra_name,
        sensitive_registry=registry,
    )

    try:
        receipt = sanitize_file(
            input_path=args.input_path,
            output_path=args.output_path,
            receipt_path=args.receipt_path,
            sanitizer=sanitizer,
        )
        print("Sanitization successful!")
        print(f"  Input:         {args.input_path}")
        print(f"  Output:        {args.output_path}")
        print(f"  Receipt ID:    {receipt.receipt_id}")
        print(f"  Receipt Hash:  {receipt.receipt_hash}")
        print(f"  Input SHA256:  {receipt.input_hash}")
        print(f"  Output SHA256: {receipt.output_hash}")
        print(f"  Findings:      {receipt.findings_count}")
        print(f"  Clean:         {receipt.is_clean}")
        return 0
    except Exception as exc:
        print(f"Error during sanitization: {exc}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

