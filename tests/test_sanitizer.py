"""Comprehensive unit test suite for PII & Legal Compliance Auditor.

Tests all requirements:
1. Automated PII scrubbing and redaction with deterministic placeholder tokens.
2. License verification checking permissive licenses and rejecting copyleft / proprietary sources.
3. Cryptographic audit receipts, deep object recursion, and edge cases.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from goldtrace_mint.factory.sanitizer import (
    DEFAULT_REDACTION_TOKENS,
    ComplianceAuditor,
    LicenseCategory,
    LicenseVerificationResult,
    LicenseVerifier,
    PIICategory,
    PIIFinding,
    PIISanitizer,
    RepositoryLicenseReport,
    SanitizationReceipt,
    SensitivePhraseEntry,
    SensitivePhraseRegistry,
    canonical_json,
    main as sanitizer_main,
    redact_pii,
    sanitize_file,
    scan_pii,
    sha256_bytes,
    sha256_json,
    sha256_text,
    verify_license,
    verify_repository_license,
)


# ============================================================================
# 1. PII SCRUBBING & REDACTION TESTS
# ============================================================================

class TestPIIScrubbingAndRedaction:
    """Test suite for automated PII and secret detection & deterministic redaction."""

    @pytest.fixture
    def sanitizer(self) -> PIISanitizer:
        return PIISanitizer()

    # --- Email Redaction ---
    @pytest.mark.parametrize(
        "raw_text,expected_contains",
        [
            ("Contact alice@example.com for info.", "Contact [REDACTED_EMAIL] for info."),
            ("Send to john.doe+support@sub.domain.co.uk immediately.", "Send to [REDACTED_EMAIL] immediately."),
            ("Emails: user1@test.org and user2@test.net", "Emails: [REDACTED_EMAIL] and [REDACTED_EMAIL]"),
        ],
    )
    def test_email_redaction(self, sanitizer: PIISanitizer, raw_text: str, expected_contains: str) -> None:
        redacted, findings = sanitizer.redact_text(raw_text)
        assert redacted == expected_contains
        assert any(f.category == PIICategory.EMAIL for f in findings)
        assert "[REDACTED_EMAIL]" in redacted

    # --- Phone Number Redaction ---
    @pytest.mark.parametrize(
        "raw_text,expected_contains",
        [
            ("Call me at 123-456-7890 today.", "Call me at [REDACTED_PHONE] today."),
            ("Office: (555) 123-4567.", "Office: [REDACTED_PHONE]."),
            ("Direct line: +1-800-555-0199.", "Direct line: [REDACTED_PHONE]."),
            ("UK Office: +44 20 7946 0958 please.", "UK Office: [REDACTED_PHONE] please."),
        ],
    )
    def test_phone_redaction(self, sanitizer: PIISanitizer, raw_text: str, expected_contains: str) -> None:
        redacted, findings = sanitizer.redact_text(raw_text)
        assert redacted == expected_contains
        assert any(f.category == PIICategory.PHONE for f in findings)
        assert "[REDACTED_PHONE]" in redacted

    # --- IP Address Redaction (IPv4 & IPv6) ---
    @pytest.mark.parametrize(
        "raw_text,expected_contains",
        [
            ("Connecting to 192.168.1.1 on port 8080", "Connecting to [REDACTED_IP_ADDRESS] on port 8080"),
            ("Public DNS: 8.8.8.8 and 1.1.1.1", "Public DNS: [REDACTED_IP_ADDRESS] and [REDACTED_IP_ADDRESS]"),
            ("Loopback is 127.0.0.1", "Loopback is [REDACTED_IP_ADDRESS]"),
            ("IPv6 target: 2001:0db8:85a3:0000:0000:8a2e:0370:7334", "IPv6 target: [REDACTED_IP_ADDRESS]"),
            ("Local IPv6: fe80::1ff:fe23:4567:890a", "Local IPv6: [REDACTED_IP_ADDRESS]"),
        ],
    )
    def test_ip_address_redaction(self, sanitizer: PIISanitizer, raw_text: str, expected_contains: str) -> None:
        redacted, findings = sanitizer.redact_text(raw_text)
        assert redacted == expected_contains
        assert any(f.category == PIICategory.IP_ADDRESS for f in findings)
        assert "[REDACTED_IP_ADDRESS]" in redacted

    # --- API Keys & Credentials Redaction ---
    def test_openai_api_key_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = "OpenAI key: sk-proj-abc1234567890def1234567890abcdef"
        redacted, findings = sanitizer.redact_text(raw)
        assert redacted == "OpenAI key: [REDACTED_API_KEY]"
        assert any(f.category == PIICategory.API_KEY for f in findings)

    def test_anthropic_api_key_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = "Anthropic key: sk-ant-api03-abcdef1234567890abcdef1234567890"
        redacted, findings = sanitizer.redact_text(raw)
        assert redacted == "Anthropic key: [REDACTED_API_KEY]"
        assert any(f.category == PIICategory.API_KEY for f in findings)

    def test_aws_credentials_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = "AWS Access Key: AKIAIOSFODNN7EXAMPLE in profile"
        redacted, findings = sanitizer.redact_text(raw)
        assert redacted == "AWS Access Key: [REDACTED_API_KEY] in profile"

    def test_github_token_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = "GH Token: ghp_1234567890abcdef1234567890abcdef123456"
        redacted, findings = sanitizer.redact_text(raw)
        assert redacted == "GH Token: [REDACTED_API_KEY]"

    def test_slack_token_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = "Slack bot: xoxb-1234567890-1234567890-abcdef1234567890123456"
        redacted, findings = sanitizer.redact_text(raw)
        assert redacted == "Slack bot: [REDACTED_API_KEY]"

    def test_jwt_token_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = "Auth header: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        redacted, findings = sanitizer.redact_text(raw)
        assert "[REDACTED_JWT]" in redacted

    def test_private_key_block_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = (
            "Here is the server key:\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Y1+examplePEMKeyBlockDataLine1\n"
            "MIIEowIBAAKCAQEA0Y1+examplePEMKeyBlockDataLine2\n"
            "-----END RSA PRIVATE KEY-----\n"
            "Keep it safe."
        )
        redacted, findings = sanitizer.redact_text(raw)
        assert "[REDACTED_PRIVATE_KEY]" in redacted
        assert "MIIEowIBAAKCAQEA0Y1" not in redacted
        assert any(f.category == PIICategory.PRIVATE_KEY for f in findings)

    def test_generic_assignments_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = 'api_key = "mySecretApiKey1234567890"\nsecret_key: "topSecretValue1234567890"\npassword="SuperSecretPassword123!"'
        redacted, findings = sanitizer.redact_text(raw)
        assert "[REDACTED_API_KEY]" in redacted
        assert "[REDACTED_SECRET]" in redacted
        assert "[REDACTED_PASSWORD]" in redacted
        assert "mySecretApiKey1234567890" not in redacted
        assert "SuperSecretPassword123!" not in redacted

    def test_bearer_token_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = "Authorization: Bearer mySecretToken1234567890abcdef12345"
        redacted, findings = sanitizer.redact_text(raw)
        assert "[REDACTED_CREDENTIAL]" in redacted
        assert "mySecretToken1234567890abcdef12345" not in redacted

    # --- Personal Names ---
    def test_honorific_name_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = "Consultation with Dr. Gregory House and Prof. Charles Xavier."
        redacted, findings = sanitizer.redact_text(raw)
        assert "[REDACTED_NAME]" in redacted
        assert "Gregory House" not in redacted
        assert "Charles Xavier" not in redacted

    def test_labeled_name_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = "Author: Alice Walker\nSigned-off-by: Bob Smith"
        redacted, findings = sanitizer.redact_text(raw)
        assert "[REDACTED_NAME]" in redacted
        assert "Alice Walker" not in redacted
        assert "Bob Smith" not in redacted

    def test_custom_registered_name(self, sanitizer: PIISanitizer) -> None:
        sanitizer.register_custom_name("Jane Doe")
        raw = "The lead engineer is Jane Doe on this project."
        redacted, findings = sanitizer.redact_text(raw)
        assert redacted == "The lead engineer is [REDACTED_NAME] on this project."
        assert any(f.rule == "custom_name" for f in findings)

    # --- SSN, Credit Card, Paths ---
    def test_ssn_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = "SSN on file: 123-45-6789."
        redacted, findings = sanitizer.redact_text(raw)
        assert redacted == "SSN on file: [REDACTED_SSN]."
        assert any(f.category == PIICategory.SSN for f in findings)

    def test_credit_card_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = "Payment card: 4111-2222-3333-4444."
        redacted, findings = sanitizer.redact_text(raw)
        assert redacted == "Payment card: [REDACTED_CREDIT_CARD]."

    def test_user_path_redaction(self, sanitizer: PIISanitizer) -> None:
        raw = "Logs at /home/john_doe/workspace/app.log and /Users/jane/config.json"
        redacted, findings = sanitizer.redact_text(raw)
        assert "[REDACTED_PATH]" in redacted
        assert "/home/john_doe" not in redacted
        assert "/Users/jane" not in redacted

    # --- Determinism & Idempotency ---
    def test_idempotency_and_determinism(self, sanitizer: PIISanitizer) -> None:
        raw = (
            "Contact user@example.com or 555-123-4567. "
            "Server at 10.0.0.1 with key sk-proj-1234567890abcdef1234567890"
        )
        pass1, findings1 = sanitizer.redact_text(raw)
        pass2, findings2 = sanitizer.redact_text(pass1)
        # Second pass should make no further changes
        assert pass1 == pass2
        assert len(findings2) == 0
        # SHA-256 hashes must be strictly identical across repeated runs
        assert sha256_text(pass1) == sha256_text(pass2)


# ============================================================================
# 2. NESTED DATA OBJECT SANITIZATION & CRYPTOGRAPHIC RECEIPTS
# ============================================================================

class TestObjectSanitizationAndReceipts:
    """Test recursive object sanitization and audit receipt generation."""

    @pytest.fixture
    def sanitizer(self) -> PIISanitizer:
        return PIISanitizer()

    def test_deep_nested_structure_redaction(self, sanitizer: PIISanitizer) -> None:
        nested = {
            "user": {
                "name_field": "Author: John Doe",
                "email": "john.doe@enterprise.com",
                "phone": "+1-800-555-0199",
                "ips": ["192.168.1.50", "10.0.0.5"],
            },
            "secrets": [
                {"provider": "openai", "token": "sk-proj-1234567890abcdef1234567890"},
                {"provider": "aws", "key": "AKIAIOSFODNN7EXAMPLE"},
            ],
            "public_metadata": {
                "version": "1.0.0",
                "count": 42,
                "is_active": True,
                "none_val": None,
            },
        }

        sanitized, findings = sanitizer.redact_object(nested)

        assert sanitized["user"]["email"] == "[REDACTED_EMAIL]"
        assert sanitized["user"]["phone"] == "[REDACTED_PHONE]"
        assert sanitized["user"]["ips"] == ["[REDACTED_IP_ADDRESS]", "[REDACTED_IP_ADDRESS]"]
        assert sanitized["secrets"][0]["token"] == "[REDACTED_API_KEY]"
        assert sanitized["secrets"][1]["key"] == "[REDACTED_API_KEY]"
        # Non-string primitives must be preserved intact
        assert sanitized["public_metadata"]["version"] == "1.0.0"
        assert sanitized["public_metadata"]["count"] == 42
        assert sanitized["public_metadata"]["is_active"] is True
        assert sanitized["public_metadata"]["none_val"] is None

    def test_sanitize_trace_receipt_generation(self, sanitizer: PIISanitizer) -> None:
        trace = {
            "trace_id": "tr-001",
            "messages": [
                {"role": "user", "content": "My email is test@company.com and IP is 172.16.0.1"},
                {"role": "assistant", "content": "Received. Using API key sk-proj-1234567890abcdef1234567890"},
            ],
            "metadata": {"source": "arena_benchmark"},
        }

        sanitized_trace, receipt = sanitizer.sanitize_trace(trace, trace_id_key="trace_id")

        assert receipt.receipt_id == "receipt-tr-001"
        assert receipt.is_clean is True
        assert receipt.findings_count == 3
        assert receipt.findings_by_category["EMAIL"] == 1
        assert receipt.findings_by_category["IP_ADDRESS"] == 1
        assert receipt.findings_by_category["API_KEY"] == 1
        assert receipt.input_hash != receipt.output_hash
        assert len(receipt.receipt_hash) == 64
        assert "test@company.com" not in json.dumps(sanitized_trace)


# ============================================================================
# 3. LEGAL LICENSE VERIFICATION TESTS
# ============================================================================

class TestLicenseVerification:
    """Test legal license verification: accept permissive, reject copyleft & proprietary."""

    @pytest.fixture
    def verifier(self) -> LicenseVerifier:
        return LicenseVerifier()

    # --- Permissive Licenses (Approved) ---
    @pytest.mark.parametrize(
        "license_id",
        [
            "MIT",
            "mit",
            "MIT-0",
            "Apache-2.0",
            "Apache 2.0",
            "APACHE-2.0",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "BSD 3-Clause",
            "0BSD",
            "Unlicense",
            "The Unlicense",
            "CC0-1.0",
            "CC0",
            "ISC",
            "Python-2.0",
        ],
    )
    def test_permissive_licenses_accepted(self, verifier: LicenseVerifier, license_id: str) -> None:
        result = verifier.verify_license_identifier(license_id)
        assert result.is_compliant is True, f"Expected {license_id} to be compliant"
        assert result.category == LicenseCategory.PERMISSIVE
        assert result.confidence == 1.0

    # --- Copyleft Licenses (Rejected) ---
    @pytest.mark.parametrize(
        "license_id",
        [
            "GPL-2.0",
            "GPL-3.0",
            "GPLv3",
            "GPLv2",
            "GPL-3.0-only",
            "GPL-3.0-or-later",
            "AGPL-3.0",
            "AGPLv3",
            "LGPL-2.1",
            "LGPL-3.0",
            "LGPLv3",
            "SSPL-1.0",
            "EUPL-1.2",
            "MPL-2.0",
        ],
    )
    def test_copyleft_licenses_rejected(self, verifier: LicenseVerifier, license_id: str) -> None:
        result = verifier.verify_license_identifier(license_id)
        assert result.is_compliant is False, f"Expected copyleft {license_id} to be rejected"
        assert result.category == LicenseCategory.COPYLEFT
        assert "copyleft" in result.reason.lower() or "rejected" in result.reason.lower()

    # --- Proprietary & Restricted Licenses (Rejected) ---
    @pytest.mark.parametrize(
        "license_str",
        [
            "All Rights Reserved",
            "Proprietary and Confidential",
            "Commercial License Required",
            "Unauthorized copying of this file is strictly prohibited",
            "Non-Commercial Use Only",
            "No License Granted",
        ],
    )
    def test_proprietary_terms_rejected(self, verifier: LicenseVerifier, license_str: str) -> None:
        result = verifier.verify_license_identifier(license_str)
        assert result.is_compliant is False
        assert result.category == LicenseCategory.PROPRIETARY
        assert "proprietary" in result.reason.lower() or "rejected" in result.reason.lower()

    # --- Missing & Unknown (Fail-Closed) ---
    @pytest.mark.parametrize(
        "unknown_license",
        ["", "   ", "CustomMadeLicense-99", "UnknownRandomTerms"],
    )
    def test_fail_closed_on_unknown(self, verifier: LicenseVerifier, unknown_license: str) -> None:
        result = verifier.verify_license_identifier(unknown_license)
        assert result.is_compliant is False
        assert result.category == LicenseCategory.UNKNOWN

    # --- Full License Text Identification ---
    def test_mit_license_text_detection(self, verifier: LicenseVerifier) -> None:
        mit_text = (
            "MIT License\n\n"
            "Copyright (c) 2026 GoldTrace Contributors\n\n"
            "Permission is hereby granted, free of charge, to any person obtaining a copy "
            "of this software and associated documentation files, to deal in the Software without restriction. "
            "The above copyright notice and this permission notice shall be included in all copies."
        )
        res = verifier.verify_license_text(mit_text)
        assert res.is_compliant is True
        assert res.category == LicenseCategory.PERMISSIVE
        assert res.license_id == "MIT"

    def test_apache2_license_text_detection(self, verifier: LicenseVerifier) -> None:
        apache_text = (
            "Apache License, Version 2.0\n"
            "Licensed under the Apache License, Version 2.0 (the \"License\");\n"
            "you may not use this file except in compliance with the License.\n"
            "You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0"
        )
        res = verifier.verify_license_text(apache_text)
        assert res.is_compliant is True
        assert res.category == LicenseCategory.PERMISSIVE
        assert res.license_id == "Apache-2.0"

    def test_unlicense_text_detection(self, verifier: LicenseVerifier) -> None:
        unlicense_text = (
            "This is free and unencumbered software released into the public domain.\n"
            "Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software."
        )
        res = verifier.verify_license_text(unlicense_text)
        assert res.is_compliant is True
        assert res.category == LicenseCategory.PERMISSIVE
        assert res.license_id == "Unlicense"

    def test_gpl_license_text_detection(self, verifier: LicenseVerifier) -> None:
        gpl_text = (
            "GNU GENERAL PUBLIC LICENSE\n"
            "Version 3, 29 June 2007\n"
            "Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>\n"
            "Everyone is permitted to copy and distribute verbatim copies of this license document."
        )
        res = verifier.verify_license_text(gpl_text)
        assert res.is_compliant is False
        assert res.category == LicenseCategory.COPYLEFT
        assert res.license_id == "GPL-3.0"

    # --- SPDX Comment Header Extraction ---
    def test_spdx_header_extraction(self, verifier: LicenseVerifier) -> None:
        source_permissive = (
            "// SPDX-License-Identifier: MIT\n"
            "// Copyright (c) 2026\n"
            "function main() { return 0; }\n"
        )
        res_permissive = verifier.verify_license_text(source_permissive)
        assert res_permissive.is_compliant is True
        assert res_permissive.license_id == "MIT"

        source_copyleft = (
            "# SPDX-License-Identifier: GPL-3.0-or-later\n"
            "def run(): pass\n"
        )
        res_copyleft = verifier.verify_license_text(source_copyleft)
        assert res_copyleft.is_compliant is False
        assert res_copyleft.category == LicenseCategory.COPYLEFT

    # --- Source Metadata Dictionary Verification ---
    def test_verify_source_metadata_dict(self, verifier: LicenseVerifier) -> None:
        # String field
        res1 = verifier.verify_source_metadata({"license": "Apache-2.0", "repo": "goldtrace"})
        assert res1.is_compliant is True

        # Dict field
        res2 = verifier.verify_source_metadata({"license": {"type": "MIT"}})
        assert res2.is_compliant is True

        # Multi-license with copyleft
        res3 = verifier.verify_source_metadata({"license": ["MIT", "GPL-3.0"]})
        assert res3.is_compliant is False

        # Missing license
        res4 = verifier.verify_source_metadata({"repo": "no_license_repo"})
        assert res4.is_compliant is False


# ============================================================================
# 4. REPOSITORY SCANNING AUDIT TESTS
# ============================================================================

class TestRepositoryLicenseAuditing:
    """Test file-system repository license audits."""

    @pytest.fixture
    def verifier(self) -> LicenseVerifier:
        return LicenseVerifier()

    def test_compliant_mit_repo_scan(self, verifier: LicenseVerifier, tmp_path: Path) -> None:
        # Create mock repo with LICENSE file
        (tmp_path / "LICENSE").write_text(
            "MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy...",
            encoding="utf-8",
        )
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "demo"\nversion = "0.1.0"\nlicense = "MIT"\n',
            encoding="utf-8",
        )
        (tmp_path / "app.py").write_text("# SPDX-License-Identifier: MIT\nprint('hello')", encoding="utf-8")

        report = verifier.verify_repository(tmp_path)
        assert report.is_compliant is True
        assert len(report.rejected_reasons) == 0
        assert len(report.files_inspected) >= 2
        assert len(report.report_hash) == 64

    def test_rejected_copyleft_repo_scan(self, verifier: LicenseVerifier, tmp_path: Path) -> None:
        # Create mock repo with GPL COPYING file
        (tmp_path / "COPYING").write_text(
            "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\nFree Software Foundation, Inc.",
            encoding="utf-8",
        )
        (tmp_path / "package.json").write_text('{"name": "gpl-pkg", "license": "GPL-3.0"}', encoding="utf-8")

        report = verifier.verify_repository(tmp_path)
        assert report.is_compliant is False
        assert any("copyleft" in r.lower() for r in report.rejected_reasons)

    def test_rejected_unlicensed_repo_scan(self, verifier: LicenseVerifier, tmp_path: Path) -> None:
        # Empty repo with no license files
        (tmp_path / "main.py").write_text("print('no license')", encoding="utf-8")
        report = verifier.verify_repository(tmp_path)
        assert report.is_compliant is False
        assert any("no license" in r.lower() for r in report.rejected_reasons)


# ============================================================================
# 5. UNIFIED COMPLIANCE AUDITOR & CONVENIENCE HELPERS
# ============================================================================

class TestComplianceAuditor:
    """Test unified end-to-end compliance auditor and convenience wrappers."""

    def test_unified_auditor_compliant_record(self) -> None:
        auditor = ComplianceAuditor()
        record = {
            "id": "ingot-101",
            "author": "Author: John Doe",
            "contact": "support@enterprise.com",
            "ip": "10.0.0.42",
            "license": "Apache-2.0",
            "prompt": "Optimize SQL query on database",
        }

        result = auditor.audit_and_sanitize_record(record, source_license="Apache-2.0")

        assert result["is_compliant"] is True
        assert result["record"]["author"] == "Author: [REDACTED_NAME]"
        assert result["record"]["contact"] == "[REDACTED_EMAIL]"
        assert result["record"]["ip"] == "[REDACTED_IP_ADDRESS]"
        assert result["license_verification"]["is_compliant"] is True
        assert result["license_verification"]["license_id"] == "APACHE-2.0"
        assert result["sanitization_receipt"]["is_clean"] is True
        assert len(result["audit_hash"]) == 64

    def test_unified_auditor_copyleft_record_rejected(self) -> None:
        auditor = ComplianceAuditor()
        record = {
            "id": "ingot-102",
            "license": "GPL-3.0",
            "code": "def solve(): return 42",
        }

        result = auditor.audit_and_sanitize_record(record, source_license="GPL-3.0")

        assert result["is_compliant"] is False
        assert result["license_verification"]["is_compliant"] is False
        assert result["license_verification"]["category"] == LicenseCategory.COPYLEFT

    def test_convenience_helpers(self) -> None:
        # scan_pii
        findings = scan_pii("Contact user@example.com")
        assert len(findings) == 1
        assert findings[0]["category"] == "EMAIL"

        # redact_pii
        cleaned, red_findings = redact_pii("Key is sk-proj-1234567890abcdef1234567890")
        assert cleaned == "Key is [REDACTED_API_KEY]"
        assert len(red_findings) == 1

        # verify_license
        lic_res = verify_license("MIT")
        assert lic_res["is_compliant"] is True

        lic_gpl = verify_license("GPL-3.0")
        assert lic_gpl["is_compliant"] is False


# ============================================================================
# 6. SENSITIVE PHRASE REGISTRY TESTS
# ============================================================================

# No phrases ship built in, so the suite registers its own neutral fixtures.
FIXTURE_ALIASES = ["Kestrel", "Nightjar", "Wren", "Pip", "Sable"]
FIXTURE_SIGNAL_PHRASES = [
    "Tides",
    "Tidepools",
    "Velvet Harbor",
    "Copper Lantern",
    "Quiet Signal",
    "Amber Hour",
    "North Window",
    "Paper Crane",
    "Glass Orchard",
    "Salt Road",
]
FIXTURE_ENTITIES = ["Project Nightjar", "Project Kestrel", "Entity Alpha"]


def fixture_registry() -> SensitivePhraseRegistry:
    registry = SensitivePhraseRegistry()
    for alias in FIXTURE_ALIASES:
        registry.register_alias(alias)
    for phrase in FIXTURE_SIGNAL_PHRASES:
        registry.register_signal_phrase(phrase)
    for entity in FIXTURE_ENTITIES:
        registry.register_entity(entity)
    return registry


class TestSensitivePhraseRegistry:
    """Test suite for SensitivePhraseRegistry: aliases, entities, signal phrases."""

    @pytest.fixture
    def registry(self) -> SensitivePhraseRegistry:
        return fixture_registry()

    @pytest.fixture
    def sanitizer(self) -> PIISanitizer:
        return PIISanitizer(sensitive_registry=fixture_registry())

    def test_registry_starts_empty(self) -> None:
        """No phrases ship with the package; a bare sanitizer redacts none."""
        assert len(SensitivePhraseRegistry()) == 0
        redacted, findings = PIISanitizer().redact_text("Kestrel walked the Salt Road.")
        assert redacted == "Kestrel walked the Salt Road."
        assert findings == []

    def test_signal_phrase_redaction(self, sanitizer: PIISanitizer) -> None:
        """Registered signal phrases redact to [SYNTHETIC_SIGNAL]."""
        text = (
            "She counted the Tides and mapped the Tidepools. "
            "He anchored at Velvet Harbor and lit the Copper Lantern at night."
        )
        redacted, findings = sanitizer.redact_text(text)

        assert "[SYNTHETIC_SIGNAL]" in redacted
        assert "Tides" not in redacted
        assert "Tidepools" not in redacted
        assert "Velvet Harbor" not in redacted
        assert "Copper Lantern" not in redacted
        assert any(f.matched_text == "Tides" for f in findings)
        assert any(f.matched_text == "Tidepools" for f in findings)
        assert any(f.matched_text == "Velvet Harbor" for f in findings)
        assert any(f.matched_text == "Copper Lantern" for f in findings)

    def test_alias_redaction(self, sanitizer: PIISanitizer) -> None:
        """Registered aliases redact to [REDACTED_ALIAS]."""
        text = "Message from Kestrel to Nightjar: Hey Wren, Pip, Sable, are you there?"
        redacted, findings = sanitizer.redact_text(text)

        assert "[REDACTED_ALIAS]" in redacted
        for alias in FIXTURE_ALIASES:
            assert alias not in redacted
        assert any(f.matched_text == "Kestrel" and f.replacement == "[REDACTED_ALIAS]" for f in findings)
        assert any(f.matched_text == "Nightjar" and f.replacement == "[REDACTED_ALIAS]" for f in findings)

    def test_multi_word_signal_phrases_case_insensitive(self, sanitizer: PIISanitizer) -> None:
        """Multi-word phrases match regardless of case."""
        text = (
            "They sent a quiet signal during the amber hour. "
            "The north window faced a paper crane over the glass orchard and the salt road."
        )
        redacted, findings = sanitizer.redact_text(text)

        assert "quiet signal" not in redacted.lower()
        assert "amber hour" not in redacted.lower()
        assert "north window" not in redacted.lower()
        assert "paper crane" not in redacted.lower()
        assert "glass orchard" not in redacted.lower()
        assert "salt road" not in redacted.lower()
        assert redacted.count("[SYNTHETIC_SIGNAL]") >= 6

    def test_custom_entity_redaction(self, sanitizer: PIISanitizer) -> None:
        """Verify project codenames and sensitive entities redact to [REDACTED_ENTITY]."""
        text = "Operation Project Nightjar is synchronized with Project Kestrel and Entity Alpha."
        redacted, findings = sanitizer.redact_text(text)

        assert "[REDACTED_ENTITY]" in redacted
        assert "Project Nightjar" not in redacted
        assert "Project Kestrel" not in redacted
        assert "Entity Alpha" not in redacted
        assert any(f.category == PIICategory.ENTITY for f in findings)

    def test_custom_phrase_registration(self, registry: SensitivePhraseRegistry) -> None:
        """Verify registering custom phrases with custom replacement tokens and categories."""
        registry.register_phrase("CustomSignalPhrase", replacement="[SYNTHETIC_SIGNAL]", category=PIICategory.SYNTHETIC_SIGNAL)
        registry.register_phrase("CodenamedValkyrie", replacement="[REDACTED_ENTITY]", category=PIICategory.ENTITY)
        registry.register_phrase("MyNicknameAlias", replacement="[REDACTED_ALIAS]", category=PIICategory.ALIAS)

        sanitizer = PIISanitizer(sensitive_registry=registry)
        text = "Check CustomSignalPhrase and CodenamedValkyrie for MyNicknameAlias."
        redacted, findings = sanitizer.redact_text(text)

        assert "[SYNTHETIC_SIGNAL]" in redacted
        assert "[REDACTED_ENTITY]" in redacted
        assert "[REDACTED_ALIAS]" in redacted
        assert "CustomSignalPhrase" not in redacted
        assert "CodenamedValkyrie" not in redacted
        assert "MyNicknameAlias" not in redacted

    def test_convenience_register_helpers(self, registry: SensitivePhraseRegistry) -> None:
        """Verify register_alias, register_entity, register_signal_phrase helper methods."""
        registry.register_alias("SuperNickname")
        registry.register_entity("TopSecretProjectX")
        registry.register_signal_phrase("PrivateSignalPhrase")

        text = "SuperNickname discussed TopSecretProjectX with PrivateSignalPhrase."
        redacted, findings = registry.redact(text)

        assert "SuperNickname" not in redacted
        assert "TopSecretProjectX" not in redacted
        assert "PrivateSignalPhrase" not in redacted
        assert "[REDACTED_ALIAS]" in redacted
        assert "[REDACTED_ENTITY]" in redacted
        assert "[SYNTHETIC_SIGNAL]" in redacted

    def test_batch_registration(self, registry: SensitivePhraseRegistry) -> None:
        """Verify batch registration with lists and dictionaries."""
        # List registration
        registry.register_batch(["SecretCode1", "SecretCode2"], default_replacement="[REDACTED_ALIAS]")
        assert "SecretCode1" in registry
        assert "SecretCode2" in registry

        # Dict registration
        registry.register_batch({
            "PhraseAlpha": "[SYNTHETIC_SIGNAL]",
            "PhraseBeta": "[REDACTED_ENTITY]",
        })
        text = "Secrets: SecretCode1, PhraseAlpha, and PhraseBeta."
        redacted, _ = registry.redact(text)
        assert "SecretCode1" not in redacted
        assert "PhraseAlpha" not in redacted
        assert "PhraseBeta" not in redacted

    def test_remove_and_clear_registry(self, registry: SensitivePhraseRegistry) -> None:
        """Verify removing individual phrases and clearing the entire registry."""
        assert "Kestrel" in registry
        removed = registry.remove_phrase("Kestrel")
        assert removed is True
        assert "Kestrel" not in registry

        registry.clear()
        assert len(registry) == 0
        assert "Nightjar" not in registry

    def test_word_boundaries_and_case_insensitivity(self, sanitizer: PIISanitizer) -> None:
        """Verify word boundaries prevent accidental substring corruption (e.g. 'Tides' in 'riptides')."""
        text = "The report cited riptides, but she watched the tides and the tidepools at ebbtide."
        redacted, findings = sanitizer.redact_text(text)

        # 'riptides' and 'ebbtide' must NOT be corrupted
        assert "riptides" in redacted
        assert "ebbtide" in redacted
        # 'tides' and 'tidepools' as standalone words MUST be redacted
        assert re.search(r"\btides\b", redacted, re.IGNORECASE) is None
        assert re.search(r"\btidepools\b", redacted, re.IGNORECASE) is None

    def test_longer_match_precedence(self, sanitizer: PIISanitizer) -> None:
        """Verify that the longer registered phrase wins (Tidepools vs Tides)."""
        text = "Look at the Tidepools."
        redacted, findings = sanitizer.redact_text(text)

        assert redacted == "Look at the [SYNTHETIC_SIGNAL]."
        assert len(findings) == 1
        assert findings[0].matched_text == "Tidepools"


# ============================================================================
# 7. PATH REDACTION: OPERATOR HOME -> /workspace/... TESTS
# ============================================================================

class TestOperatorHomePathRedaction:
    """Path redaction converting the operator's own home to /workspace/...

    The home is pinned explicitly rather than taken from the running user, so
    these assertions describe the sanitiser instead of describing the machine
    the suite happens to run on.
    """

    OPERATOR_HOME = "/home/abcd"

    @pytest.fixture
    def sanitizer(self) -> PIISanitizer:
        return PIISanitizer(workspace_home=self.OPERATOR_HOME, sensitive_registry=fixture_registry())

    def test_operator_home_dots_converted_to_workspace(self, sanitizer: PIISanitizer) -> None:
        """Verify /home/abcd/... converts cleanly to /workspace/..."""
        text = "Repository is cloned at /home/abcd/..."
        redacted, findings = sanitizer.redact_text(text)
        assert redacted == "Repository is cloned at /workspace/..."
        assert any(f.rule == "workspace_home_path" and f.replacement == "/workspace/..." for f in findings)

    def test_operator_home_deep_subpath_conversion(self, sanitizer: PIISanitizer) -> None:
        """Verify deep file path /home/abcd/Desktop/GTDataworks/... converts to /workspace/Desktop/..."""
        text = "Script located at /home/abcd/Desktop/GTDataworks/3.Dataworks-Refinery/src/goldtrace_mint/factory/sanitizer.py"
        redacted, findings = sanitizer.redact_text(text)
        assert redacted == "Script located at /workspace/Desktop/GTDataworks/3.Dataworks-Refinery/src/goldtrace_mint/factory/sanitizer.py"
        assert "/home/abcd" not in redacted

    def test_operator_home_root_conversion(self, sanitizer: PIISanitizer) -> None:
        """Verify root /home/abcd converts to /workspace"""
        text = "User home directory: /home/abcd"
        redacted, findings = sanitizer.redact_text(text)
        assert redacted == "User home directory: /workspace"

    def test_other_user_paths_redacted_to_placeholder(self, sanitizer: PIISanitizer) -> None:
        """Verify non-abcd user paths (/home/john_doe, /Users/jane) are redacted to [REDACTED_PATH]."""
        text = "Logs at /home/john_doe/app.log and macOS config at /Users/jane/settings.json"
        redacted, findings = sanitizer.redact_text(text)
        assert redacted == "Logs at [REDACTED_PATH] and macOS config at [REDACTED_PATH]"
        assert "/home/john_doe" not in redacted
        assert "/Users/jane" not in redacted

    def test_workspace_rewrite_follows_the_configured_operator(self) -> None:
        """The rewrite belongs to whoever is running it, not to one username.

        The home was hardcoded to /home/abcd, so on any other machine the
        structure-preserving rewrite silently did nothing and every operator
        path fell through to [REDACTED_PATH].
        """
        sanitizer = PIISanitizer(workspace_home="/home/dana")
        redacted, findings = sanitizer.redact_text("Ran /home/dana/run.sh")
        assert redacted == "Ran /workspace/run.sh"
        assert any(f.rule == "workspace_home_path" for f in findings)

    def test_a_different_operators_home_is_still_opaque(self) -> None:
        """Only the configured operator is rewritten; everyone else is redacted."""
        sanitizer = PIISanitizer(workspace_home="/home/dana")
        redacted, _ = sanitizer.redact_text("Ran /home/abcd/run.sh")
        assert redacted == "Ran [REDACTED_PATH]"
        assert "/home/abcd" not in redacted

    def test_a_longer_username_sharing_a_prefix_is_not_rewritten(self) -> None:
        """/home/abcdef must not be treated as the operator's own /home/abcd."""
        sanitizer = PIISanitizer(workspace_home="/home/abcd")
        redacted, _ = sanitizer.redact_text("Ran /home/abcdef/run.sh")
        assert redacted == "Ran [REDACTED_PATH]"

    def test_mixed_paths_and_pii_in_single_text(self, sanitizer: PIISanitizer) -> None:
        """Verify text containing /home/abcd/..., other user paths, emails, and sensitive phrases."""
        text = (
            "User Kestrel ran /home/abcd/run.sh with email admin@example.com, "
            "referencing /home/bob/data.csv and saying Tides."
        )
        redacted, findings = sanitizer.redact_text(text)

        assert redacted == "User [REDACTED_ALIAS] ran /workspace/run.sh with email [REDACTED_EMAIL], referencing [REDACTED_PATH] and saying [SYNTHETIC_SIGNAL]."
        assert "/home/abcd" not in redacted
        assert "/home/bob" not in redacted
        assert "admin@example.com" not in redacted
        assert "Kestrel" not in redacted
        assert "Tides" not in redacted


# ============================================================================
# 8. STAND-ALONE BATCH SANITIZATION CLI & RECEIPT TESTS
# ============================================================================

class TestSanitizeFileBatch:
    """Test suite for sanitize_file batch processor on JSON, JSONL, and text files."""

    def test_sanitize_json_file(self, tmp_path: Path) -> None:
        """Verify sanitize_file on a structured raw JSON file."""
        input_file = tmp_path / "raw_input.json"
        output_file = tmp_path / "sanitized_output.json"
        receipt_file = tmp_path / "custom_receipt.json"

        raw_data = {
            "developer": "Author: Alice Walker",
            "email": "alice@security.corp",
            "api_key": "sk-proj-1234567890abcdef1234567890",
            "repo_path": "/home/abcd/Desktop/GTDataworks/project",
            "notes": "Kestrel mentioned Tides and Velvet Harbor during the call",
            "count": 100,
            "is_active": True,
        }
        input_file.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")

        receipt = sanitize_file(
            input_path=input_file,
            output_path=output_file,
            receipt_path=receipt_file,
            sanitizer=PIISanitizer(workspace_home="/home/abcd", sensitive_registry=fixture_registry()),
        )

        assert output_file.exists()
        assert receipt_file.exists()
        assert receipt.is_clean is True
        assert receipt.findings_count >= 5
        assert receipt.input_hash != receipt.output_hash
        assert len(receipt.receipt_hash) == 64

        # Verify output content is valid JSON and clean
        out_data = json.loads(output_file.read_text(encoding="utf-8"))
        assert out_data["developer"] == "Author: [REDACTED_NAME]"
        assert out_data["email"] == "[REDACTED_EMAIL]"
        assert out_data["api_key"] == "[REDACTED_API_KEY]"
        assert out_data["repo_path"] == "/workspace/Desktop/GTDataworks/project"
        assert "[REDACTED_ALIAS]" in out_data["notes"]
        assert "[SYNTHETIC_SIGNAL]" in out_data["notes"]
        assert "Kestrel" not in json.dumps(out_data)
        assert "Tides" not in json.dumps(out_data)
        assert "alice@security.corp" not in json.dumps(out_data)

        # Verify receipt file content
        stored_receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        assert stored_receipt["receipt_id"] == receipt.receipt_id
        assert stored_receipt["receipt_hash"] == receipt.receipt_hash
        assert stored_receipt["input_hash"] == receipt.input_hash
        assert stored_receipt["output_hash"] == receipt.output_hash

    def test_receipt_records_file_names_not_directories(self, tmp_path: Path) -> None:
        """A redaction receipt must not leak the operator's directory layout."""
        input_file = tmp_path / "raw.txt"
        input_file.write_text("contact admin@example.com", encoding="utf-8")
        receipt = sanitize_file(input_file, tmp_path / "clean.txt")

        assert receipt.metadata["input_file"] == "raw.txt"
        assert receipt.metadata["output_file"] == "clean.txt"
        assert str(tmp_path) not in json.dumps(receipt.to_dict())

    def test_sanitize_jsonl_file(self, tmp_path: Path) -> None:
        """Verify sanitize_file on a JSONL / NDJSON dataset file."""
        input_file = tmp_path / "traces.jsonl"
        output_file = tmp_path / "traces_clean.jsonl"

        lines = [
            json.dumps({"id": 1, "msg": "Call me at +1-800-555-0199", "alias": "Nightjar", "path": "/home/abcd/..."}),
            json.dumps({"id": 2, "msg": "Secret sk-ant-api03-abcdef1234567890abcdef1234567890", "term": "Tidepools"}),
            json.dumps({"id": 3, "msg": "Server IP 192.168.1.100", "phrase": "Copper Lantern"}),
        ]
        input_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        receipt = sanitize_file(
            input_path=input_file,
            output_path=output_file,
            sanitizer=PIISanitizer(workspace_home="/home/abcd", sensitive_registry=fixture_registry()),
        )

        assert output_file.exists()
        assert receipt.is_clean is True
        assert receipt.findings_count >= 6

        # Auto-generated receipt file should exist
        default_rcpt = Path(f"{output_file}.receipt.json")
        assert default_rcpt.exists()

        out_lines = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(out_lines) == 3
        assert out_lines[0]["msg"] == "Call me at [REDACTED_PHONE]"
        assert out_lines[0]["alias"] == "[REDACTED_ALIAS]"
        assert out_lines[0]["path"] == "/workspace/..."
        assert out_lines[1]["term"] == "[SYNTHETIC_SIGNAL]"
        assert out_lines[2]["phrase"] == "[SYNTHETIC_SIGNAL]"

    def test_sanitize_text_file(self, tmp_path: Path) -> None:
        """Verify sanitize_file on a plain text / log file."""
        input_file = tmp_path / "raw_log.txt"
        output_file = tmp_path / "clean_log.txt"
        sanitizer = PIISanitizer(workspace_home="/home/abcd", sensitive_registry=fixture_registry())

        content = (
            "2026-08-08 12:00:00 [INFO] User Kestrel connected from 10.0.0.1\n"
            "2026-08-08 12:00:05 [DEBUG] Loaded configuration from /home/abcd/config.yaml\n"
            "2026-08-08 12:00:10 [INFO] Mentioned Tides and Quiet Signal in session\n"
        )
        input_file.write_text(content, encoding="utf-8")

        receipt = sanitize_file(
            input_path=input_file,
            output_path=output_file,
            sanitizer=sanitizer,
        )

        assert output_file.exists()
        assert receipt.is_clean is True
        cleaned_content = output_file.read_text(encoding="utf-8")

        assert "Kestrel" not in cleaned_content
        assert "10.0.0.1" not in cleaned_content
        assert "/home/abcd" not in cleaned_content
        assert "/workspace/config.yaml" in cleaned_content
        assert "Tides" not in cleaned_content
        assert "Quiet Signal" not in cleaned_content
        assert "[REDACTED_ALIAS]" in cleaned_content
        assert "[REDACTED_IP_ADDRESS]" in cleaned_content
        assert "[SYNTHETIC_SIGNAL]" in cleaned_content

    def test_sanitize_file_nonexistent_raises(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError on nonexistent input."""
        with pytest.raises(FileNotFoundError):
            sanitize_file(tmp_path / "missing.json", tmp_path / "out.json")

    def test_cli_main_execution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify CLI main entrypoint executes cleanly and parses args."""
        in_file = tmp_path / "cli_in.txt"
        out_file = tmp_path / "cli_out.txt"
        rcpt_file = tmp_path / "cli_rcpt.json"

        in_file.write_text("Secret Kestrel at /home/abcd/test.py with SuperCustomPhrase", encoding="utf-8")

        test_args = [
            "goldtrace-sanitizer",
            str(in_file),
            str(out_file),
            "--receipt-path",
            str(rcpt_file),
            "--extra-name",
            "SpecialUser",
            "--extra-phrase",
            "Kestrel",
            "--extra-phrase",
            "SuperCustomPhrase",
        ]
        monkeypatch.setattr("sys.argv", test_args)
        # Pin the operator home the default sanitizer resolves, so the
        # /workspace/... assertion describes the sanitiser, not this machine.
        monkeypatch.setattr(Path, "home", lambda: Path("/home/abcd"))

        exit_code = sanitizer_main()
        assert exit_code == 0
        assert out_file.exists()
        assert rcpt_file.exists()

        out_text = out_file.read_text(encoding="utf-8")
        assert "Secret [REDACTED_ALIAS] at /workspace/test.py with [REDACTED_ALIAS]" in out_text

