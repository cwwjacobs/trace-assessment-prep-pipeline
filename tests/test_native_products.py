from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from goldtrace_refinery import native_products
from goldtrace_refinery.native_products import (
    MANIFEST_CONTRACT,
    PROJECTION_RULESET_SHA256,
    ROW_CONTRACT,
    NativeTelemetryPackError,
    produce_native_telemetry_pack,
    verify_native_telemetry_pack,
)
from goldtrace_refinery.paths import ensure_labyrinth_importable
from goldtrace_refinery.release_privacy import (
    RELEASE_PRIVACY_RECEIPT_SCHEMA,
    RELEASE_PRIVACY_RECEIPT_SCHEMA_V1,
    RELEASE_PRIVACY_RULESET_IDENTITY,
    RELEASE_PRIVACY_RULESET_IDENTITY_V1,
)


ensure_labyrinth_importable()
# Native-trace capture lives in the Labyrinth (Tier 1) package, which is not
# part of this repository. Skip rather than fail when it is not installed.
pytest.importorskip("goldentrace", reason="requires the Labyrinth `goldentrace` package")

from goldentrace.capture import verify_event_stream  # noqa: E402
from goldentrace.native_trace_capture import capture_native_trace  # noqa: E402
from goldentrace.native_trace_contract import (  # noqa: E402
    ADAPTER_REGISTRY,
    AGY_ADAPTER_ID,
    CLAUDE_ADAPTER_ID,
    CODEX_ADAPTER_ID,
    GROK_JSONL_ADAPTER_ID,
    GROK_JSON_ADAPTER_ID,
    CHATGPT_EXPORT_ADAPTER_ID,
    CLAUDE_EXPORT_ADAPTER_ID,
)


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_BODY = "private-payload-never-export"
PRIVATE_SESSION = "private-session-native-identifier"
PRIVATE_TOOL = "private-tool-data-never-export"
PRIVATE_REASONING = "private-reasoning-never-export"
FROZEN_PORTABLE_V1_RULESET_SHA256 = (
    "080995bf28e4237797b87807b92d0d1fd338dd66e2a3bb2539360c300c67e094"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _line(value: object, *, final_lf: bool = True) -> bytes:
    return _canonical(value) + (b"\n" if final_lf else b"")


def _samples() -> dict[str, bytes]:
    return {
        AGY_ADAPTER_ID: b"".join(
            [
                _line(
                    {
                        "type": "USER_INPUT",
                        "text": PRIVATE_BODY,
                        "session_id": PRIVATE_SESSION,
                    }
                ),
                _line(
                    {
                        "type": "RUN_COMMAND",
                        "command": PRIVATE_TOOL,
                    }
                ),
            ]
        ),
        CLAUDE_ADAPTER_ID: b"".join(
            [
                _line(
                    {
                        "type": "user",
                        "message": {"content": PRIVATE_BODY},
                        "uuid": PRIVATE_SESSION,
                    }
                ),
                _line(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {"type": "thinking", "thinking": PRIVATE_REASONING},
                                {"type": "text", "text": PRIVATE_BODY},
                            ]
                        },
                    }
                ),
            ]
        ),
        CODEX_ADAPTER_ID: b"".join(
            [
                _line(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": PRIVATE_TOOL,
                            "arguments": PRIVATE_BODY,
                            "call_id": PRIVATE_SESSION,
                        },
                    }
                ),
                _line(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "reasoning",
                            "summary": PRIVATE_REASONING,
                        },
                    }
                ),
                _line(
                    {
                        "type": "session_meta",
                        "payload": {"id": PRIVATE_SESSION},
                    }
                ),
            ]
        ),
        GROK_JSONL_ADAPTER_ID: b"".join(
            [
                _line({"type": "assistant", "content": PRIVATE_BODY}),
                _line({"type": "reasoning", "content": PRIVATE_REASONING}),
                _line(
                    {
                        "type": "backend_tool_call",
                        "name": PRIVATE_TOOL,
                        "arguments": PRIVATE_BODY,
                    },
                    final_lf=False,
                ),
            ]
        ),
        GROK_JSON_ADAPTER_ID: _canonical(
            {
                "session": PRIVATE_SESSION,
                "content": PRIVATE_BODY,
                "reasoning": PRIVATE_REASONING,
                "tool": PRIVATE_TOOL,
            }
        ),
        # Web-export conversation lanes are held opaque exactly like the grok
        # JSON lane: one conversation document per source file.
        CHATGPT_EXPORT_ADAPTER_ID: _canonical(
            {
                "title": PRIVATE_SESSION,
                "mapping": {PRIVATE_SESSION: {"content": PRIVATE_BODY}},
            }
        ),
        CLAUDE_EXPORT_ADAPTER_ID: _canonical(
            {
                "name": PRIVATE_SESSION,
                "conversation": PRIVATE_BODY,
            }
        ),
    }


def _capture(tmp_path: Path, adapter_id: str, source: bytes | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = source if source is not None else _samples()[adapter_id]
    suffix = ".json" if adapter_id == GROK_JSON_ADAPTER_ID else ".jsonl"
    source_path = tmp_path / f"private-source-name-{len(list(tmp_path.iterdir()))}{suffix}"
    source_path.write_bytes(payload)
    result = capture_native_trace(
        source_path,
        runs_root=tmp_path / "runs",
        adapter_id=adapter_id,
    )
    return result, source_path


def _pack_bytes(pack: Path) -> bytes:
    return b"\n".join(
        path.read_bytes()
        for path in sorted(pack.rglob("*"))
        if path.is_file()
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rewrite_inventory(pack: Path, relative: str) -> None:
    manifest_path = pack / "MANIFEST.json"
    manifest = _load_json(manifest_path)
    payload = (pack / relative).read_bytes()
    for item in manifest["files"]:
        if item["name"] == relative:
            item["size_bytes"] = len(payload)
            item["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_bytes(_canonical(manifest))
    manifest_path.chmod(0o600)


def _historical_v1_clean_receipt(row: dict[str, Any]) -> dict[str, Any]:
    row_id = row["row_id"]
    assert isinstance(row_id, str)
    row_sha256 = hashlib.sha256(_canonical(row)).hexdigest()
    receipt = {
        "schema_id": RELEASE_PRIVACY_RECEIPT_SCHEMA_V1,
        "record_id_sha256": hashlib.sha256(row_id.encode("utf-8")).hexdigest(),
        "ruleset_identity": RELEASE_PRIVACY_RULESET_IDENTITY_V1,
        "ruleset_sha256": FROZEN_PORTABLE_V1_RULESET_SHA256,
        "input_row_sha256": row_sha256,
        "output_row_sha256": row_sha256,
        "findings_count": 0,
        "findings_by_category": {},
        "output_changed": False,
        "residual_findings_count": 0,
        "residual_findings_by_category": {},
        "status": "PASS_AUTOMATED_SCAN",
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return receipt


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


def test_contract_copies_are_byte_identical_and_valid() -> None:
    for filename in (MANIFEST_CONTRACT, ROW_CONTRACT):
        canonical = (COMPONENT_ROOT / "contracts" / filename).read_bytes()
        packaged = (
            COMPONENT_ROOT / "src" / "goldtrace_refinery" / "contracts" / filename
        ).read_bytes()
        assert canonical == packaged
        Draft202012Validator.check_schema(json.loads(canonical))

    manifest_schema = json.loads(
        (COMPONENT_ROOT / "contracts" / MANIFEST_CONTRACT).read_text(encoding="utf-8")
    )
    assert (
        manifest_schema["properties"]["projection_ruleset"]["properties"]["sha256"][
            "const"
        ]
        == PROJECTION_RULESET_SHA256
    )
    assert (
        native_products.PROJECTION_RULESET_DOCUMENT["privacy_boundary"]
        == RELEASE_PRIVACY_RULESET_IDENTITY
    )


@pytest.mark.parametrize("adapter_id", tuple(ADAPTER_REGISTRY))
def test_every_closed_registered_adapter_produces_only_private_hold_telemetry(
    tmp_path: Path,
    adapter_id: str,
) -> None:
    assert set(_samples()) == set(ADAPTER_REGISTRY)
    result, source_path = _capture(tmp_path, adapter_id)
    source_before = source_path.read_bytes()
    pack = tmp_path / "telemetry-pack"

    exported = produce_native_telemetry_pack(result.bundle_path, pack)
    verified = verify_native_telemetry_pack(pack, source_bundle=result.bundle_path)

    assert exported.provider == ADAPTER_REGISTRY[adapter_id].provider
    assert exported.adapter == adapter_id
    assert verified.provider == exported.provider
    assert verified.adapter == adapter_id
    assert verified.disposition == "HOLD"
    assert verified.training_readiness == "NOT_TRAINING_READY"
    assert verified.row_count == len(_samples()[adapter_id].splitlines())
    assert source_path.read_bytes() == source_before

    assert stat.S_IMODE(pack.stat().st_mode) == 0o700
    assert stat.S_IMODE((pack / "provenance").stat().st_mode) == 0o700
    for path in pack.rglob("*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert path.stat().st_nlink == 1

    manifest = _load_json(pack / "MANIFEST.json")
    rows = _load_jsonl(pack / "telemetry.jsonl")
    receipts = _load_jsonl(pack / "provenance" / "privacy-receipts.jsonl")
    assert manifest["source"] == {
        "provider": ADAPTER_REGISTRY[adapter_id].provider,
        "adapter": adapter_id,
        "capture_mode": ADAPTER_REGISTRY[adapter_id].capture_mode,
        "adapter_spec_sha256": ADAPTER_REGISTRY[adapter_id].spec_sha256,
        "source_manifest_sha256": result.verification.manifest_sha256,
        "event_head_sha256": result.verification.event_head_sha256,
        "native_trace_index_sha256": _load_json(result.bundle_path / "run.json")[
            "native_trace_index"
        ]["sha256"],
        "selected_source_sha256": result.source_sha256,
        "archive_sha256": None,
    }
    assert manifest["governance"] == {
        "visibility": "PRIVATE",
        "rights_status": "UNKNOWN",
        "disposition": "HOLD",
        "release_admitted": False,
        "training_readiness": "NOT_TRAINING_READY",
        "reasoning_policy": "EXCLUDE",
    }
    assert manifest["counts"]["row_count"] == len(rows)
    assert manifest["counts"]["privacy_receipt_count"] == len(rows)
    assert all(receipt["schema_id"] == RELEASE_PRIVACY_RECEIPT_SCHEMA for receipt in receipts)
    assert all(receipt["coverage"]["input"]["complete"] is True for receipt in receipts)
    assert all(
        receipt["coverage"]["residual_output"]["complete"] is True
        for receipt in receipts
    )
    for row in rows:
        assert row["governance"] == manifest["governance"]
        assert row["semantic_status"] == "NATIVE_TELEMETRY_ONLY_NOT_EXECUTION_WITNESS"
        assert "messages" not in row
        assert "native_record" not in row

    emitted = _pack_bytes(pack)
    for private_value in (
        PRIVATE_BODY,
        PRIVATE_SESSION,
        PRIVATE_TOOL,
        PRIVATE_REASONING,
        result.run_id,
        source_path.name,
        str(source_path),
    ):
        assert private_value.encode("utf-8") not in emitted
    events = verify_event_stream(
        result.bundle_path / "events" / "events.ndjson",
        expected_run_id=result.run_id,
        expected_classification="native_trace_import",
    )
    for event in events:
        assert event.event_id.encode("utf-8") not in emitted

    pack_documents: list[object] = [manifest, *rows]
    pack_documents.extend(_load_jsonl(pack / "provenance" / "lineage.jsonl"))
    pack_documents.extend(receipts)
    keys = _all_keys(pack_documents)
    assert {
        "run_id",
        "session_id",
        "event_id",
        "member",
        "member_name",
        "path",
        "content",
        "tool_data",
        "native_record",
    }.isdisjoint(keys)


def test_unknown_and_malformed_records_remain_opaque_hold_telemetry(tmp_path: Path) -> None:
    source = b"".join(
        [
            _line({"type": "USER_INPUT", "text": PRIVATE_BODY}),
            _line({"type": "FUTURE_UNKNOWN", "value": PRIVATE_BODY}),
            b'{"type":"USER_INPUT","broken":\n',
        ]
    )
    result, _ = _capture(tmp_path, AGY_ADAPTER_ID, source)
    pack = tmp_path / "opaque-pack"

    produce_native_telemetry_pack(result.bundle_path, pack)
    rows = _load_jsonl(pack / "telemetry.jsonl")

    assert len(rows) == 3
    assert rows[1]["semantic_kind"] == "unknown"
    assert rows[1]["supported_discriminator"] is None
    assert rows[1]["projection_available"] is False
    assert rows[2]["malformed"] is True
    assert rows[2]["projection_available"] is False
    assert all(row["governance"]["disposition"] == "HOLD" for row in rows)
    assert PRIVATE_BODY.encode() not in _pack_bytes(pack)


def test_zip_member_metadata_and_paths_never_cross_product_boundary(tmp_path: Path) -> None:
    trace = _line(
        {
            "type": "USER_INPUT",
            "text": PRIVATE_BODY,
            "session_id": PRIVATE_SESSION,
        }
    )
    archive = tmp_path / "private-archive-name.zip"
    member_name = "private/member/session/events.jsonl"
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(member_name, trace)
        handle.writestr("unselected/private.txt", PRIVATE_BODY)
    archive_bytes = archive.read_bytes()
    result = capture_native_trace(
        archive,
        runs_root=tmp_path / "runs",
        adapter_id=AGY_ADAPTER_ID,
        zip_member=member_name,
    )
    pack = tmp_path / "zip-telemetry"

    produce_native_telemetry_pack(result.bundle_path, pack)

    emitted = _pack_bytes(pack)
    assert member_name.encode() not in emitted
    assert archive.name.encode() not in emitted
    assert PRIVATE_BODY.encode() not in emitted
    manifest = _load_json(pack / "MANIFEST.json")
    assert manifest["source"]["archive_sha256"] == hashlib.sha256(archive_bytes).hexdigest()
    assert "member" not in manifest["source"]
    assert archive.read_bytes() == archive_bytes


def test_same_source_produces_byte_identical_packs(tmp_path: Path) -> None:
    result, _ = _capture(tmp_path, CODEX_ADAPTER_ID)
    first = tmp_path / "first"
    second = tmp_path / "second"

    produce_native_telemetry_pack(result.bundle_path, first)
    produce_native_telemetry_pack(result.bundle_path, second)

    assert {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }


def test_producer_refuses_to_overwrite_even_an_empty_output_directory(tmp_path: Path) -> None:
    result, _ = _capture(tmp_path, AGY_ADAPTER_ID)
    output = tmp_path / "existing"
    output.mkdir(mode=0o700)
    marker = output / "operator-marker"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(NativeTelemetryPackError, match="already exists"):
        produce_native_telemetry_pack(result.bundle_path, output)

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_unsupported_atomic_rename_uses_fail_closed_nonoverwrite_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _capture(tmp_path, AGY_ADAPTER_ID)
    output = tmp_path / "fallback-pack"
    monkeypatch.setattr(
        native_products,
        "_rename_directory_noreplace",
        lambda _source, _destination: False,
    )

    exported = produce_native_telemetry_pack(result.bundle_path, output)
    verified = verify_native_telemetry_pack(output, source_bundle=result.bundle_path)

    assert exported.output_path == output
    assert verified.disposition == "HOLD"
    assert not (output / ".INSTALLING").exists()


def test_verifier_rejects_claim_widening_even_after_inventory_rehash(tmp_path: Path) -> None:
    result, _ = _capture(tmp_path, AGY_ADAPTER_ID)
    pack = tmp_path / "claim-widened"
    produce_native_telemetry_pack(result.bundle_path, pack)
    telemetry = pack / "telemetry.jsonl"
    rows = _load_jsonl(telemetry)
    rows[0]["run_id"] = result.run_id
    rows[0]["content"] = PRIVATE_BODY
    telemetry.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))
    telemetry.chmod(0o600)
    _rewrite_inventory(pack, "telemetry.jsonl")

    with pytest.raises(NativeTelemetryPackError):
        verify_native_telemetry_pack(pack, source_bundle=result.bundle_path)


def test_verifier_rejects_registry_tuple_claim_widening(tmp_path: Path) -> None:
    result, _ = _capture(tmp_path, AGY_ADAPTER_ID)
    pack = tmp_path / "registry-widened"
    produce_native_telemetry_pack(result.bundle_path, pack)
    manifest_path = pack / "MANIFEST.json"
    manifest = _load_json(manifest_path)
    manifest["source"]["provider"] = "claude"
    manifest_path.write_bytes(_canonical(manifest))
    manifest_path.chmod(0o600)

    with pytest.raises(NativeTelemetryPackError):
        verify_native_telemetry_pack(pack, source_bundle=result.bundle_path)


def test_verifier_rejects_privacy_receipt_tamper_after_inventory_rehash(tmp_path: Path) -> None:
    result, _ = _capture(tmp_path, GROK_JSONL_ADAPTER_ID)
    pack = tmp_path / "receipt-tampered"
    produce_native_telemetry_pack(result.bundle_path, pack)
    receipts_path = pack / "provenance" / "privacy-receipts.jsonl"
    receipts = _load_jsonl(receipts_path)
    receipts[0]["findings_count"] = 1
    receipts[0]["findings_by_category"] = {"EMAIL": 1}
    receipts_path.write_bytes(b"".join(_canonical(row) + b"\n" for row in receipts))
    receipts_path.chmod(0o600)
    _rewrite_inventory(pack, "provenance/privacy-receipts.jsonl")

    with pytest.raises(NativeTelemetryPackError):
        verify_native_telemetry_pack(pack, source_bundle=result.bundle_path)


def test_historical_v1_pack_verifies_without_regenerating_current_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _capture(tmp_path, AGY_ADAPTER_ID)
    pack = tmp_path / "historical-v1-pack"
    produce_native_telemetry_pack(result.bundle_path, pack)

    rows = _load_jsonl(pack / "telemetry.jsonl")
    receipts = [_historical_v1_clean_receipt(row) for row in rows]
    receipts_path = pack / "provenance" / "privacy-receipts.jsonl"
    receipts_path.write_bytes(b"".join(_canonical(receipt) + b"\n" for receipt in receipts))
    receipts_path.chmod(0o600)

    lineage_path = pack / "provenance" / "lineage.jsonl"
    lineage = _load_jsonl(lineage_path)
    for item, receipt in zip(lineage, receipts, strict=True):
        item["privacy_receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
        item["privacy_ruleset_sha256"] = receipt["ruleset_sha256"]
    lineage_path.write_bytes(b"".join(_canonical(item) + b"\n" for item in lineage))
    lineage_path.chmod(0o600)
    _rewrite_inventory(pack, "provenance/privacy-receipts.jsonl")
    _rewrite_inventory(pack, "provenance/lineage.jsonl")

    def _must_not_regenerate(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("historical verification regenerated a current receipt")

    monkeypatch.setattr(native_products, "sanitize_release_row", _must_not_regenerate)

    verified = verify_native_telemetry_pack(pack, source_bundle=result.bundle_path)

    assert verified.disposition == "HOLD"
    assert verified.training_readiness == "NOT_TRAINING_READY"


def test_verifier_rejects_wrong_source_bundle(tmp_path: Path) -> None:
    first_result, _ = _capture(
        tmp_path / "first-source",
        AGY_ADAPTER_ID,
        _line({"type": "USER_INPUT", "text": "synthetic-first"}),
    )
    second_result, _ = _capture(
        tmp_path / "second-source",
        AGY_ADAPTER_ID,
        _line({"type": "USER_INPUT", "text": "synthetic-second"}),
    )
    pack = tmp_path / "source-bound-pack"
    produce_native_telemetry_pack(first_result.bundle_path, pack)

    with pytest.raises(NativeTelemetryPackError):
        verify_native_telemetry_pack(pack, source_bundle=second_result.bundle_path)


@pytest.mark.parametrize("mutation", ["extra_file", "public_mode", "symlink"])
def test_verifier_rejects_unsafe_or_extra_pack_topology(
    tmp_path: Path,
    mutation: str,
) -> None:
    result, _ = _capture(tmp_path, CLAUDE_ADAPTER_ID)
    pack = tmp_path / "unsafe-pack"
    produce_native_telemetry_pack(result.bundle_path, pack)

    if mutation == "extra_file":
        extra = pack / "unexpected.json"
        extra.write_bytes(b"{}")
        extra.chmod(0o600)
    elif mutation == "public_mode":
        (pack / "telemetry.jsonl").chmod(0o644)
    else:
        os.symlink("telemetry.jsonl", pack / "unexpected-link")

    with pytest.raises(NativeTelemetryPackError):
        verify_native_telemetry_pack(pack, source_bundle=result.bundle_path)


@pytest.mark.parametrize("reader_kind", ["pack", "source"])
@pytest.mark.parametrize("replacement_kind", ["regular", "symlink"])
def test_descriptor_bound_read_rejects_lstat_open_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader_kind: str,
    replacement_kind: str,
) -> None:
    root = tmp_path / "pack"
    root.mkdir(mode=0o700)
    target = root / "telemetry.jsonl"
    target.write_bytes(b"original\n")
    target.chmod(0o600)
    displaced = root / "held-original"
    replacement = root / "replacement"
    replacement.write_bytes(b"widened!\n")
    replacement.chmod(0o600)
    real_open = os.open
    swapped = False

    def swap_before_open(path: os.PathLike[str] | str, flags: int, *args: int) -> int:
        nonlocal swapped
        if not swapped and Path(path) == target:
            swapped = True
            target.rename(displaced)
            if replacement_kind == "regular":
                replacement.rename(target)
            else:
                target.symlink_to(displaced.name)
        return real_open(path, flags, *args)

    monkeypatch.setattr(native_products.os, "open", swap_before_open)

    message = "pack file is unsafe" if reader_kind == "pack" else "source file is unsafe"
    with pytest.raises(NativeTelemetryPackError, match=message):
        if reader_kind == "pack":
            native_products._read_pack_file(
                root,
                "telemetry.jsonl",
                max_bytes=1024,
            )
        else:
            native_products._read_regular_source_file(target, max_bytes=1024)

    assert swapped is True


def test_source_truth_rejects_bundle_root_rebinding_after_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _capture(tmp_path, AGY_ADAPTER_ID)
    original_root = result.bundle_path
    displaced_root = original_root.with_name(f"{original_root.name}-held")
    real_source_truth = native_products._source_truth_from_bound_root

    def swap_after_verification(bound_root: Path):
        truth = real_source_truth(bound_root)
        bound_root.rename(displaced_root)
        bound_root.mkdir(mode=0o700)
        return truth

    monkeypatch.setattr(
        native_products,
        "_source_truth_from_bound_root",
        swap_after_verification,
    )

    with pytest.raises(NativeTelemetryPackError, match="source root binding changed"):
        native_products._source_truth(original_root)


def test_verifier_rejects_pack_root_rebinding_after_exact_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _capture(tmp_path, AGY_ADAPTER_ID)
    pack = tmp_path / "telemetry-pack"
    produce_native_telemetry_pack(result.bundle_path, pack)
    displaced_pack = tmp_path / "telemetry-pack-held"
    real_bound_verifier = native_products._verify_native_telemetry_pack_bound

    def swap_after_exact_reads(
        bound_root: Path,
        *,
        first_truth: Any,
    ) -> Any:
        verification = real_bound_verifier(bound_root, first_truth=first_truth)
        bound_root.rename(displaced_pack)
        bound_root.mkdir(mode=0o700)
        return verification

    monkeypatch.setattr(
        native_products,
        "_verify_native_telemetry_pack_bound",
        swap_after_exact_reads,
    )

    with pytest.raises(NativeTelemetryPackError, match="pack root binding changed"):
        verify_native_telemetry_pack(pack, source_bundle=result.bundle_path)
