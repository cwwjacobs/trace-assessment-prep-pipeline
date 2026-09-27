from __future__ import annotations

import hashlib
import json
import string
from typing import Any

import pytest

from goldtrace_refinery.release_privacy import (
    RELEASE_PRIVACY_RECEIPT_SCHEMA,
    RELEASE_PRIVACY_RECEIPT_SCHEMA_V1,
    RELEASE_PRIVACY_RULESET_IDENTITY,
    RELEASE_PRIVACY_RULESET_IDENTITY_V1,
    ReleasePrivacyError,
    release_ruleset_sha256,
    sanitize_release_row,
    verify_release_privacy_receipt,
    verify_release_privacy_receipt_integrity,
)


WORKSPACE_HOME = "/home/release-operator"
FROZEN_PORTABLE_V1_RULESET_SHA256 = (
    "080995bf28e4237797b87807b92d0d1fd338dd66e2a3bb2539360c300c67e094"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _rehash(receipt: dict[str, Any]) -> None:
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()


def _frozen_v1_receipt() -> tuple[dict[str, Any], dict[str, Any]]:
    row = {"prompt": "Historical public fixture."}
    receipt = {
        "schema_id": RELEASE_PRIVACY_RECEIPT_SCHEMA_V1,
        "record_id_sha256": (
            "e42c58247506398ad723afda7db8fe079a0447b74492ee0c31a291dcff0f0aae"
        ),
        "ruleset_identity": RELEASE_PRIVACY_RULESET_IDENTITY_V1,
        "ruleset_sha256": FROZEN_PORTABLE_V1_RULESET_SHA256,
        "input_row_sha256": (
            "3e05877b20c0c58531d8bd0c34823d49283668c071e80ed0c42433c821d72896"
        ),
        "output_row_sha256": (
            "3e05877b20c0c58531d8bd0c34823d49283668c071e80ed0c42433c821d72896"
        ),
        "findings_count": 0,
        "findings_by_category": {},
        "output_changed": False,
        "residual_findings_count": 0,
        "residual_findings_by_category": {},
        "status": "PASS_AUTOMATED_SCAN",
        "receipt_sha256": (
            "0c91e7f25b1ba98a5d2d8d8b9b3199dd5fcf375a9bc8534b314ce399fd397e5a"
        ),
    }
    return row, receipt


def test_prompt_only_secret_is_redacted_and_bound_to_exact_canonical_rows() -> None:
    secret = "sk-proj-1234567890abcdef1234567890"
    row = {"prompt": f"Use only this credential: {secret}", "temperature": 0}

    output, receipt = sanitize_release_row(
        row,
        record_id="record-prompt-only",
        workspace_home=WORKSPACE_HOME,
    )

    assert output == {"prompt": "Use only this credential: [REDACTED_API_KEY]", "temperature": 0}
    assert receipt["schema_id"] == RELEASE_PRIVACY_RECEIPT_SCHEMA
    assert receipt["ruleset_identity"] == RELEASE_PRIVACY_RULESET_IDENTITY
    assert receipt["status"] == "PASS_AUTOMATED_REDACTION"
    assert receipt["findings_count"] == 1
    assert receipt["findings_by_category"] == {"API_KEY": 1}
    assert receipt["output_changed"] is True
    assert receipt["input_row_sha256"] == hashlib.sha256(_canonical(row)).hexdigest()
    assert receipt["output_row_sha256"] == hashlib.sha256(_canonical(output)).hexdigest()
    assert verify_release_privacy_receipt(
        receipt,
        output,
        workspace_home=WORKSPACE_HOME,
    ) == []


def test_assistant_content_and_nested_tool_arguments_are_both_scanned() -> None:
    email = "person@example.test"
    bearer = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
    row = {
        "messages": [
            {"role": "assistant", "content": f"Send the result to {email}."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "request",
                        "arguments": {"headers": {"authorization": bearer}},
                    }
                ],
            },
        ]
    }

    output, receipt = sanitize_release_row(
        row,
        record_id="record-assistant-tool",
        workspace_home=WORKSPACE_HOME,
    )

    rendered = json.dumps(output, sort_keys=True)
    assert email not in rendered
    assert bearer not in rendered
    assert "[REDACTED_EMAIL]" in rendered
    assert "[REDACTED_CREDENTIAL]" in rendered
    assert receipt["findings_by_category"] == {"CREDENTIAL": 1, "EMAIL": 1}
    assert verify_release_privacy_receipt(
        receipt,
        output,
        workspace_home=WORKSPACE_HOME,
    ) == []


def test_clean_row_has_deterministic_scan_receipt_without_timestamp() -> None:
    row = {"messages": [{"role": "user", "content": "Sort these public integers."}]}

    first = sanitize_release_row(
        row,
        record_id="record-clean",
        workspace_home=WORKSPACE_HOME,
    )
    second = sanitize_release_row(
        row,
        record_id="record-clean",
        workspace_home=WORKSPACE_HOME,
    )

    assert first == second
    output, receipt = first
    assert output == row
    assert receipt["status"] == "PASS_AUTOMATED_SCAN"
    assert receipt["findings_count"] == 0
    assert receipt["findings_by_category"] == {}
    assert receipt["output_changed"] is False
    assert "timestamp" not in receipt
    assert receipt["ruleset_sha256"] == release_ruleset_sha256(workspace_home=WORKSPACE_HOME)
    assert receipt["coverage"]["scope"] == "all_json_string_values_and_object_keys"
    assert receipt["coverage"]["input"]["complete"] is True
    assert receipt["coverage"]["residual_output"]["complete"] is True
    assert receipt["coverage"]["input"]["object_key_strings_scanned"] == 3
    assert receipt["coverage"]["input"]["string_values_scanned"] == 2
    assert receipt["coverage"]["input"] == receipt["coverage"]["residual_output"]


def test_default_release_ruleset_is_host_independent_and_redacts_real_home_paths() -> None:
    row = {"prompt": "Read /home/alice/private/session.json before answering."}

    output, receipt = sanitize_release_row(row, record_id="record-portable")

    assert output == {"prompt": "Read [REDACTED_PATH] before answering."}
    assert receipt["ruleset_sha256"] == release_ruleset_sha256()
    assert verify_release_privacy_receipt(receipt, output) == []


def test_high_entropy_fallback_covers_unknown_value_and_object_key_tokens() -> None:
    value_token = "hf_" + string.ascii_letters[:40]
    key_token = "unrecognized_" + string.ascii_letters + string.digits
    row = {
        "nested": [{"credential": value_token}],
        key_token: "public-value",
    }

    output, receipt = sanitize_release_row(row, record_id="record-entropy")

    rendered = json.dumps(output, sort_keys=True)
    assert value_token not in rendered
    assert key_token not in rendered
    assert output["nested"][0]["credential"] == "[REDACTED_SECRET]"
    assert output["[REDACTED_SECRET]"] == "public-value"
    assert receipt["findings_by_category"] == {"SECRET": 2}
    assert receipt["coverage"]["input"]["object_key_strings_scanned"] == 3
    assert receipt["coverage"]["input"]["string_values_scanned"] == 2
    assert verify_release_privacy_receipt(receipt, output) == []


def test_long_low_entropy_text_is_not_treated_as_a_secret() -> None:
    prose = "documentation" * 12
    row = {"prompt": prose}

    output, receipt = sanitize_release_row(row, record_id="record-low-entropy")

    assert output == row
    assert receipt["findings_count"] == 0
    assert verify_release_privacy_receipt(receipt, output) == []


def test_structured_sha256_allowance_is_field_scoped() -> None:
    digest = hashlib.sha256(b"synthetic-public-digest").hexdigest()
    row = {
        "event_head_sha256": digest,
        "sha256": digest,
        "row_id": "ntelem-" + digest,
        "credential_sha256": digest,
        "untyped": digest,
    }

    output, receipt = sanitize_release_row(row, record_id="record-structured-digest")

    assert output["event_head_sha256"] == digest
    assert output["sha256"] == digest
    assert output["row_id"] == "ntelem-" + digest
    assert output["credential_sha256"] == "[REDACTED_SECRET]"
    assert output["untyped"] == "[REDACTED_SECRET]"
    assert receipt["findings_by_category"] == {"SECRET": 2}
    assert verify_release_privacy_receipt(receipt, output) == []


def test_high_entropy_fallback_covers_base64url_hyphen_tokens() -> None:
    token = "ABcd1234EF-ghIJ5678KL-mnOP9012QR-stUV3456WX"

    output, receipt = sanitize_release_row(
        {"credential": token},
        record_id="record-base64url-entropy",
    )

    assert output == {"credential": "[REDACTED_SECRET]"}
    assert receipt["findings_by_category"] == {"SECRET": 1}
    assert verify_release_privacy_receipt(receipt, output) == []


def test_empty_row_declares_complete_zero_text_coverage() -> None:
    output, receipt = sanitize_release_row({}, record_id="record-empty")

    expected_phase = {
        "object_key_strings_total": 0,
        "object_key_strings_scanned": 0,
        "string_values_total": 0,
        "string_values_scanned": 0,
        "utf8_bytes_total": 0,
        "utf8_bytes_scanned": 0,
        "complete": True,
    }
    assert output == {}
    assert receipt["findings_count"] == 0
    assert receipt["coverage"]["input"] == expected_phase
    assert receipt["coverage"]["residual_output"] == expected_phase
    assert verify_release_privacy_receipt(receipt, output) == []


def test_receipt_contains_no_matched_plaintext_previews_or_record_identifier() -> None:
    secret = "password=SuperSecretPassword123!"
    record_id = "record-person@example.test"
    output, receipt = sanitize_release_row(
        {"assistant": secret},
        record_id=record_id,
        workspace_home=WORKSPACE_HOME,
    )

    receipt_text = json.dumps(receipt, sort_keys=True)
    assert "SuperSecretPassword123!" not in receipt_text
    assert "SuperSe" not in receipt_text
    assert "person@example.test" not in receipt_text
    assert "matched" not in receipt_text
    assert "preview" not in receipt_text
    assert "path" not in receipt_text
    assert output == {"assistant": "password=[REDACTED_PASSWORD]"}


def test_output_and_receipt_tampering_fail_even_with_recomputed_self_hash() -> None:
    output, receipt = sanitize_release_row(
        {"prompt": "Contact first@example.test"},
        record_id="record-tamper",
        workspace_home=WORKSPACE_HOME,
    )

    tampered_output = {"prompt": "Contact second@example.test"}
    output_errors = verify_release_privacy_receipt(
        receipt,
        tampered_output,
        workspace_home=WORKSPACE_HOME,
    )
    assert "receipt output row hash mismatch" in output_errors
    assert "output row still contains privacy findings" in output_errors

    tampered_receipt = dict(receipt)
    tampered_receipt["findings_count"] = 0
    tampered_receipt["findings_by_category"] = {}
    _rehash(tampered_receipt)
    receipt_errors = verify_release_privacy_receipt(
        tampered_receipt,
        output,
        workspace_home=WORKSPACE_HOME,
    )
    assert "receipt findings_count disagrees with output_changed" in receipt_errors


def test_ruleset_mismatch_fails_even_with_recomputed_self_hash() -> None:
    output, receipt = sanitize_release_row(
        {"prompt": "No private material here."},
        record_id="record-ruleset",
        workspace_home=WORKSPACE_HOME,
    )
    tampered = dict(receipt)
    tampered["ruleset_sha256"] = "0" * 64
    _rehash(tampered)

    errors = verify_release_privacy_receipt(
        tampered,
        output,
        workspace_home=WORKSPACE_HOME,
    )

    assert "receipt ruleset hash mismatch" in errors
    assert "receipt self-hash mismatch" not in errors


def test_coverage_missing_or_tampered_fails_after_recomputed_self_hash() -> None:
    output, receipt = sanitize_release_row(
        {"prompt": "Public synthetic text."},
        record_id="record-coverage-tamper",
    )

    missing = dict(receipt)
    missing.pop("coverage")
    _rehash(missing)
    missing_errors = verify_release_privacy_receipt(missing, output)
    assert "receipt is missing fields: coverage" in missing_errors
    assert "receipt coverage is malformed" in missing_errors

    tampered = json.loads(json.dumps(receipt))
    tampered["coverage"]["residual_output"]["string_values_scanned"] += 1
    _rehash(tampered)
    tampered_errors = verify_release_privacy_receipt(tampered, output)
    assert "receipt residual coverage is malformed" in tampered_errors
    assert "receipt residual coverage differs from output traversal" in tampered_errors

    missing_detector = json.loads(json.dumps(receipt))
    missing_detector["coverage"]["detectors"].pop()
    _rehash(missing_detector)
    detector_errors = verify_release_privacy_receipt(missing_detector, output)
    assert "receipt coverage detectors mismatch" in detector_errors


def test_entropy_residual_in_tampered_output_fails_closed() -> None:
    output, receipt = sanitize_release_row(
        {"prompt": "Public synthetic text."},
        record_id="record-entropy-residual",
    )
    assert output == {"prompt": "Public synthetic text."}
    tampered_output = {"prompt": "unknown_" + string.ascii_letters + string.digits}

    errors = verify_release_privacy_receipt(receipt, tampered_output)

    assert "receipt output row hash mismatch" in errors
    assert "output row still contains privacy findings" in errors
    assert "receipt residual coverage differs from output traversal" in errors


def test_frozen_v1_is_integrity_only_and_cannot_satisfy_current_admission() -> None:
    row, receipt = _frozen_v1_receipt()

    integrity = verify_release_privacy_receipt_integrity(
        receipt,
        row,
        record_id="historical-v1-fixture",
    )
    admission_errors = verify_release_privacy_receipt(
        receipt,
        row,
        record_id="historical-v1-fixture",
    )

    assert integrity.ok is True
    assert integrity.status == "HISTORICAL_INTEGRITY_ONLY"
    assert admission_errors == [
        "historical v1 receipt is integrity-only and cannot satisfy current admission"
    ]


def test_unknown_policy_tuple_fails_closed() -> None:
    output, receipt = sanitize_release_row(
        {"prompt": "Public synthetic text."},
        record_id="record-unknown-policy",
    )
    receipt["ruleset_identity"] = "goldtrace.release-privacy-ruleset.unknown"
    _rehash(receipt)

    integrity = verify_release_privacy_receipt_integrity(receipt, output)

    assert integrity.ok is False
    assert integrity.status == "INVALID"
    assert "receipt privacy policy tuple is unknown" in integrity.errors


@pytest.mark.parametrize(
    "row,record_id",
    [
        ({"bad": float("nan")}, "record-nan"),
        ({1: "ambiguous integer key"}, "record-key"),
        ({"bad": {1, 2, 3}}, "record-set"),
        ({"ok": True}, ""),
    ],
)
def test_malformed_release_input_fails_closed(row: dict[Any, Any], record_id: str) -> None:
    with pytest.raises(ReleasePrivacyError):
        sanitize_release_row(
            row,
            record_id=record_id,
            workspace_home=WORKSPACE_HOME,
        )


def test_redacted_key_collision_fails_closed() -> None:
    row = {
        "first@example.test": "one",
        "second@example.test": "two",
    }

    with pytest.raises(ReleasePrivacyError, match="collide object keys"):
        sanitize_release_row(
            row,
            record_id="record-key-collision",
            workspace_home=WORKSPACE_HOME,
        )


def test_unknown_receipt_field_is_rejected_to_keep_receipt_privacy_safe() -> None:
    output, receipt = sanitize_release_row(
        {"prompt": "No private material here."},
        record_id="record-extra",
        workspace_home=WORKSPACE_HOME,
    )
    receipt["matched_preview"] = "must never be accepted"
    _rehash(receipt)

    errors = verify_release_privacy_receipt(
        receipt,
        output,
        workspace_home=WORKSPACE_HOME,
    )

    assert "receipt has 1 unknown field(s)" in errors


def test_malformed_receipt_fails_closed_without_echoing_sensitive_key() -> None:
    output, receipt = sanitize_release_row(
        {"prompt": "No private material here."},
        record_id="record-malformed-receipt",
        workspace_home=WORKSPACE_HOME,
    )
    receipt[1] = "person@example.test"

    errors = verify_release_privacy_receipt(
        receipt,
        output,
        workspace_home=WORKSPACE_HOME,
    )

    assert errors == ["object key at $ must be a string"]
    assert "person@example.test" not in json.dumps(errors)
