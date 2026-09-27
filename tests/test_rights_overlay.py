"""Rights-overlay gate on refinery cast.

The contract under test, in one line: omitting an overlay changes nothing, an
exactly-binding valid overlay grants RIGHTS_ASSERTED and never PASSED, and every
other outcome fails closed at HOLD while still leaving evidence that the check
was requested and evaluated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from goldtrace_refinery.pack_record import rights_assertion_sha256
from goldtrace_refinery.rights_overlay import (
    RIGHTS_ASSERTED,
    LEDGER_SCHEMA,
    LEDGER_SCHEMA_V2,
    RightsOverlayError,
    evaluate_rights_overlay,
    load_rights_overlay,
)

MANIFEST_SHA = "a" * 64
OTHER_SHA = "b" * 64
CHECKPOINT = "head-0001"
EVENT_HEAD = "c" * 64
OTHER_EVENT_HEAD = "d" * 64


def _assertion(basis: str, by: str = "Operator <op@example.test>") -> dict:
    a = {"asserted_by": by, "asserted_at": "2026-08-14T00:00:00Z", "basis": basis}
    a["assertion_sha256"] = rights_assertion_sha256(a)
    return a


def _write_overlay(
    tmp_path: Path,
    entries: list[dict],
    *,
    assertions: dict[str, dict] | None = None,
    header: dict | None = None,
    bodies_name: str = "rights-assertions.v1.json",
    write_bodies: bool = True,
) -> Path:
    ledger = tmp_path / "rights-ledger.v1.jsonl"
    head = header if header is not None else {
        "record_type": "ledger_header",
        "ledger_schema": LEDGER_SCHEMA,
        "ledger_version": "1.0.0",
        "entry_count": len(entries),
    }
    lines = ([json.dumps(head)] if head else []) + [json.dumps(e) for e in entries]
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if write_bodies:
        (tmp_path / bodies_name).write_text(
            json.dumps({"assertions": assertions or {}}), encoding="utf-8"
        )
    return ledger


def _entry(
    status: str,
    assertion: dict,
    *,
    manifest=MANIFEST_SHA,
    checkpoint=CHECKPOINT,
    event_head=EVENT_HEAD,
    omit_event_head: bool = False,
) -> dict:
    return {
        "record_type": "rights_entry",
        "ledger_schema": LEDGER_SCHEMA,
        "sealed_artifact_id": "run-test",
        "sealed_artifact_dir": "run-test",
        "canonical_seal": {
            "manifest_sha256": manifest,
            "checkpoint_head": checkpoint,
            **({} if omit_event_head else {"event_head_sha256": event_head}),
        },
        "rights_status": status,
        "rights_assertion_sha256": assertion["assertion_sha256"],
        "asserted_by": assertion["asserted_by"],
        "asserted_at": assertion["asserted_at"],
        "assertion_body_ref": "sidecars/rights-assertions.v1.json",
    }


def _evaluate(
    ledger: Path, *, manifest=MANIFEST_SHA, checkpoint=CHECKPOINT, event_head=EVENT_HEAD
):
    return evaluate_rights_overlay(
        load_rights_overlay(ledger),
        manifest_sha256=manifest,
        checkpoint_head=checkpoint,
        event_head_sha256=event_head,
    )


# --- success -------------------------------------------------------------


def test_exact_match_on_both_bindings_grants_rights_asserted(tmp_path):
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a)],
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
    )
    result = _evaluate(ledger)
    assert result.evaluated is True
    assert result.granted is True
    assert result.to_record()["granted_status"] == RIGHTS_ASSERTED
    assert len(result.matched_entries) == 1


def test_several_distinct_classes_over_one_bundle_is_not_ambiguity(tmp_path):
    a1 = _assertion("operator authored")
    a2 = _assertion("machine results")
    ledger = _write_overlay(
        tmp_path,
        [
            _entry("OPERATOR_OWNED_FULL_RIGHTS", a1),
            _entry("MACHINE_RESULT_OPERATOR_OWNED", a2),
        ],
        assertions={
            "OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a1},
            "MACHINE_RESULT_OPERATOR_OWNED": {"rights_assertion": a2},
        },
    )
    result = _evaluate(ledger)
    assert result.granted is True
    assert len(result.matched_entries) == 2


def test_grant_never_reaches_passed(tmp_path):
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a)],
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
    )
    record = _evaluate(ledger).to_record()
    assert record["granted_status"] == RIGHTS_ASSERTED
    assert record["granted_status"] != "PASSED"


# --- fail closed: bindings ----------------------------------------------


def test_manifest_mismatch_fails_closed(tmp_path):
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a)],
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
    )
    result = _evaluate(ledger, manifest=OTHER_SHA)
    assert result.evaluated is True and result.granted is False
    assert "no ledger entry bound this bundle" in result.reason


def test_checkpoint_head_mismatch_fails_closed_even_when_manifest_matches(tmp_path):
    """The second binding is load-bearing; a manifest match alone must not pass."""
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a)],
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
    )
    result = _evaluate(ledger, checkpoint="head-DIFFERENT")
    assert result.granted is False


def test_a_null_side_is_never_counted_as_a_binding(tmp_path):
    """A ledger silent on checkpoint_head makes no claim about it.

    The entry is not rejected for silence, but the null side cannot count
    towards the minimum either: the grant has to rest on the two identifiers
    that are non-null on both sides.
    """
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a, checkpoint=None)],
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
    )
    result = _evaluate(ledger, checkpoint=CHECKPOINT)
    assert result.granted is True
    assert "checkpoint_head" not in result.bindings_used
    assert result.bindings_used == ["event_head_sha256", "manifest_sha256"]


def test_a_stated_checkpoint_that_disagrees_is_still_a_mismatch(tmp_path):
    """Silence is tolerated; a contradictory value is not."""
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a, checkpoint="head-OTHER")],
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
    )
    assert _evaluate(ledger, checkpoint=CHECKPOINT).granted is False


# --- two independent non-null bindings ----------------------------------


def test_null_checkpoint_on_both_sides_is_not_a_second_binding(tmp_path):
    """None == None identifies nothing and must never carry a grant."""
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a, checkpoint=None, omit_event_head=True)],
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
    )
    result = _evaluate(ledger, checkpoint=None, event_head=None)
    assert result.granted is False
    assert result.bindings_used == []
    assert "independently recomputed" in result.reason


def test_fallback_binding_missing_fails_closed(tmp_path):
    """Null checkpoint and no event head leaves only one binding."""
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a, checkpoint=None, omit_event_head=True)],
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
    )
    assert _evaluate(ledger, checkpoint=None, event_head=EVENT_HEAD).granted is False


def test_fallback_binding_mismatch_fails_closed(tmp_path):
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a, checkpoint=None)],
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
    )
    result = _evaluate(ledger, checkpoint=None, event_head=OTHER_EVENT_HEAD)
    assert result.granted is False


def test_manifest_plus_valid_fallback_grants(tmp_path):
    """The real CLI-ore shape: no checkpoint, event head carries the second bind."""
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a, checkpoint=None)],
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
    )
    result = _evaluate(ledger, checkpoint=None, event_head=EVENT_HEAD)
    assert result.granted is True
    assert result.bindings_used == ["event_head_sha256", "manifest_sha256"]


def test_receipt_records_which_bindings_were_used(tmp_path):
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a)],
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
    )
    record = _evaluate(ledger).to_record()
    assert record["minimum_bindings"] == 2
    assert "manifest_sha256" in record["bindings_used"]
    assert len(record["bindings_used"]) >= 2
    assert set(record["bindings_required"]) == {
        "manifest_sha256",
        "checkpoint_head",
        "event_head_sha256",
    }


# --- fail closed: policy -------------------------------------------------


def test_inadmissible_rights_status_fails_closed(tmp_path):
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("UNKNOWN", a)],
        assertions={"UNKNOWN": {"rights_assertion": a}},
    )
    result = _evaluate(ledger)
    assert result.granted is False
    assert result.rejected_entries[0]["rejection"].startswith("rights_status")


def test_tampered_assertion_body_fails_closed(tmp_path):
    """Bodies are keyed by recomputed hash, so swapping text breaks the link."""
    a = _assertion("operator authored")
    tampered = dict(a)
    tampered["basis"] = "something else entirely"
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a)],
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": tampered}},
    )
    result = _evaluate(ledger)
    assert result.granted is False
    assert "no assertion body matches" in result.rejected_entries[0]["rejection"]


def test_one_bad_entry_blocks_the_whole_grant(tmp_path):
    good = _assertion("operator authored")
    bad = _assertion("not admissible")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", good), _entry("UNKNOWN", bad)],
        assertions={
            "OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": good},
            "UNKNOWN": {"rights_assertion": bad},
        },
    )
    result = _evaluate(ledger)
    assert result.granted is False


def test_conflicting_assertions_for_one_status_is_ambiguous(tmp_path):
    a1 = _assertion("first basis")
    a2 = _assertion("second basis")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a1), _entry("OPERATOR_OWNED_FULL_RIGHTS", a2)],
        assertions={
            "one": {"rights_assertion": a1},
            "two": {"rights_assertion": a2},
        },
    )
    result = _evaluate(ledger)
    assert result.granted is False
    assert "ambiguous" in result.reason


# --- fail closed: unusable input ----------------------------------------


def test_missing_ledger_raises(tmp_path):
    with pytest.raises(RightsOverlayError, match="not found"):
        load_rights_overlay(tmp_path / "nope.jsonl")


def test_unparseable_ledger_raises(tmp_path):
    p = tmp_path / "rights-ledger.v1.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(RightsOverlayError, match="not valid JSON"):
        load_rights_overlay(p)


def test_wrong_schema_raises(tmp_path):
    a = _assertion("operator authored")
    entry = _entry("OPERATOR_OWNED_FULL_RIGHTS", a)
    entry["ledger_schema"] = "something.else.v9"
    ledger = _write_overlay(
        tmp_path,
        [entry],
        header={"record_type": "ledger_header", "ledger_schema": "something.else.v9", "entry_count": 1},
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
    )
    with pytest.raises(RightsOverlayError, match="unsupported rights ledger schema"):
        load_rights_overlay(ledger)


def test_header_entry_count_disagreement_raises(tmp_path):
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a)],
        header={
            "record_type": "ledger_header",
            "ledger_schema": LEDGER_SCHEMA,
            "entry_count": 99,
        },
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
    )
    with pytest.raises(RightsOverlayError, match="declares 99"):
        load_rights_overlay(ledger)


def test_missing_assertion_bodies_raises(tmp_path):
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path, [_entry("OPERATOR_OWNED_FULL_RIGHTS", a)], write_bodies=False
    )
    with pytest.raises(RightsOverlayError, match="missing"):
        load_rights_overlay(ledger)


def test_ledger_with_no_entries_raises(tmp_path):
    ledger = tmp_path / "rights-ledger.v1.jsonl"
    ledger.write_text(
        json.dumps({"record_type": "ledger_header", "ledger_schema": LEDGER_SCHEMA}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RightsOverlayError, match="no rights_entry records"):
        load_rights_overlay(ledger)


def test_load_accepts_v1_and_v2_schemas(tmp_path):
    a = _assertion("operator authored")
    v1 = _write_overlay(tmp_path, [_entry("OPERATOR_OWNED_FULL_RIGHTS", a)])
    assert load_rights_overlay(v1)["schema"] == LEDGER_SCHEMA
    v2_dir = tmp_path / "v2"
    v2_dir.mkdir()
    v2 = _write_overlay(
        v2_dir,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a)],
        header={
            "record_type": "ledger_header",
            "ledger_schema": LEDGER_SCHEMA_V2,
            "ledger_version": "2.0.0",
            "entry_count": 1,
        },
    )
    loaded = load_rights_overlay(v2)
    assert loaded["schema"] == LEDGER_SCHEMA_V2
    other = tmp_path / "other"
    other.mkdir()
    bad = _write_overlay(
        other,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a)],
        header={
            "record_type": "ledger_header",
            "ledger_schema": "gtdataworks.rights-ledger.v0",
            "entry_count": 1,
        },
    )
    with pytest.raises(RightsOverlayError, match="unsupported"):
        load_rights_overlay(bad)


def test_v2_fixture_without_event_head_does_not_grant(tmp_path):
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [
            _entry(
                "OPERATOR_OWNED_FULL_RIGHTS",
                a,
                checkpoint=None,
                omit_event_head=True,
            )
        ],
        header={
            "record_type": "ledger_header",
            "ledger_schema": LEDGER_SCHEMA_V2,
            "entry_count": 1,
        },
    )
    result = _evaluate(ledger, checkpoint=None, event_head=EVENT_HEAD)
    assert result.granted is False


def test_v2_event_head_mismatch_does_not_grant(tmp_path):
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a, checkpoint=None)],
        header={
            "record_type": "ledger_header",
            "ledger_schema": LEDGER_SCHEMA_V2,
            "entry_count": 1,
        },
    )
    result = _evaluate(ledger, checkpoint=None, event_head=OTHER_EVENT_HEAD)
    assert result.granted is False


def test_v2_fixture_grants_on_manifest_and_event_head(tmp_path):
    a = _assertion("operator authored")
    ledger = _write_overlay(
        tmp_path,
        [_entry("OPERATOR_OWNED_FULL_RIGHTS", a, checkpoint=None)],
        assertions={"OPERATOR_OWNED_FULL_RIGHTS": {"rights_assertion": a}},
        header={
            "record_type": "ledger_header",
            "ledger_schema": LEDGER_SCHEMA_V2,
            "entry_count": 1,
        },
    )
    result = _evaluate(ledger, checkpoint=None, event_head=EVENT_HEAD)
    assert result.granted is True
    assert result.to_record()["granted_status"] == RIGHTS_ASSERTED
    assert result.bindings_used == ["event_head_sha256", "manifest_sha256"]
