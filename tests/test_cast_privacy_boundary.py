from __future__ import annotations

import json

from goldtrace_refinery.cast import _mechanical_run_status, _refinery_disposition
from goldtrace_refinery.privacy import scan_obj, scan_text


def test_cast_privacy_findings_never_retain_preview_span_or_path() -> None:
    secret = "api_key=abcdefghijklmnop"
    findings = scan_obj({"nested": [{"value": secret}]})

    assert findings == [{"rule": "generic_api_key"}]
    rendered = json.dumps(findings, sort_keys=True)
    assert secret not in rendered
    assert "abcdefghijkl" not in rendered
    assert "preview" not in rendered
    assert "span" not in rendered
    assert "path" not in rendered
    assert scan_text(secret) == findings


def test_native_import_completion_is_mechanical_not_product_admission() -> None:
    run = {
        "classification": "native_trace_import",
        "import_status": "COMPLETED",
        "verification_status": "VERIFIED_WITH_GAPS",
        "seal_status": "SEALED_WITH_GAPS",
    }

    assert _mechanical_run_status(run, {}) == "COMPLETED"


def test_native_import_privacy_failure_remains_quarantined() -> None:
    run = {"classification": "native_trace_import"}

    assert _refinery_disposition(run, []) == ("HOLD", "HOLD")
    assert _refinery_disposition(run, [{"rule": "generic_api_key"}]) == (
        "QUARANTINED",
        "QUARANTINED",
    )
