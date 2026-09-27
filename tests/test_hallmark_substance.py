"""Substance checks: is there anything in this product at all?

Every integrity check in Hallmark is satisfied trivially by nothing. An empty
payload matches the hash of an empty payload, parses without a bad line,
declares a count of zero its manifest agrees with, and contains no secrets. A
product lot carrying

    e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

for both dataset splits — the SHA-256 of empty input — passed fifty checks and
was admitted to the vault. These tests pin the gate that stops it.

The lots built here are deliberately minimal. Other checks fail on them for
want of files; each test asserts only on the substance checks it is about, so
a substance regression cannot hide behind unrelated noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from goldtrace_hallmark.verify import verify_product_lot

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _chat_row(turns: int = 4, roles: tuple[str, ...] = ("user", "assistant")) -> dict[str, Any]:
    messages = []
    for index in range(turns):
        role = roles[index % len(roles)]
        messages.append({"role": role, "content": f"{role} turn {index}"})
    return {"messages": messages}


def _lot(
    tmp_path: Path,
    *,
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]] | None = None,
    substance: dict[str, Any] | None = None,
    validation_ratio: float = 0.0,
) -> Path:
    """Write the smallest lot that exercises the substance checks."""
    validation = validation or []
    root = tmp_path / "lot"
    (root / "dataset").mkdir(parents=True)
    (root / "curation").mkdir(parents=True)

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    write_jsonl(root / "dataset" / "train.jsonl", train)
    write_jsonl(root / "dataset" / "validation.jsonl", validation)
    (root / "dataset" / "manifest.json").write_text(
        json.dumps({"train_count": len(train), "validation_count": len(validation)}),
        encoding="utf-8",
    )

    rubric: dict[str, Any] = {"rubric_id": "substance-test", "rubric_version": "1"}
    if substance is not None:
        rubric["substance"] = substance
    (root / "curation" / "rubric.json").write_text(json.dumps(rubric), encoding="utf-8")

    (root / "product-spec.json").write_text(
        json.dumps(
            {
                "product_id": "substance-test",
                "product_version": "1.0.0",
                "fixture_only": True,
                "split_policy": {
                    "leakage_group_field": "split_group_id",
                    "train_ratio": 1.0 - validation_ratio,
                    "validation_ratio": validation_ratio,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "product-manifest.json").write_text(
        json.dumps({"product_id": "substance-test", "fixture_only": True}),
        encoding="utf-8",
    )
    return root


def _checks(root: Path) -> dict[str, dict[str, Any]]:
    report = verify_product_lot(root, mode="fixture")
    return {check["name"]: check for check in report["checks"]} | {
        "__status__": {"detail": report["status"]}
    }


def test_the_empty_digest_is_what_an_empty_split_actually_hashes_to(tmp_path: Path) -> None:
    """Anchors the constant these tests are named for."""
    import hashlib

    root = _lot(tmp_path, train=[])
    payload = (root / "dataset" / "train.jsonl").read_bytes()
    assert payload == b""
    assert hashlib.sha256(payload).hexdigest() == EMPTY_SHA256


def test_an_empty_train_split_fails(tmp_path: Path) -> None:
    checks = _checks(_lot(tmp_path, train=[]))
    assert checks["substance.train_not_empty"]["ok"] is False
    assert "rows=0" in checks["substance.train_not_empty"]["detail"]


def test_a_populated_train_split_passes(tmp_path: Path) -> None:
    checks = _checks(_lot(tmp_path, train=[_chat_row(), _chat_row()]))
    assert checks["substance.train_not_empty"]["ok"] is True
    assert checks["substance.train_min_records"]["ok"] is True


def test_a_substance_failure_is_a_fail_not_a_hold(tmp_path: Path) -> None:
    """An empty product is not a soft finding to be reviewed later."""
    checks = _checks(_lot(tmp_path, train=[]))
    assert checks["__status__"]["detail"] == "FAIL"


def test_a_rubric_cannot_authorise_an_empty_product(tmp_path: Path) -> None:
    """The floor of one record is not negotiable by configuration."""
    checks = _checks(_lot(tmp_path, train=[], substance={"min_train_records": 0}))
    assert checks["substance.train_not_empty"]["ok"] is False
    assert checks["substance.train_min_records"]["ok"] is False


@pytest.mark.parametrize("declared", [-5, "nonsense", None])
def test_a_malformed_rubric_threshold_falls_back_to_the_floor(
    tmp_path: Path, declared: Any
) -> None:
    checks = _checks(_lot(tmp_path, train=[], substance={"min_train_records": declared}))
    assert checks["substance.train_min_records"]["ok"] is False


def test_a_rubric_may_raise_the_record_floor(tmp_path: Path) -> None:
    checks = _checks(
        _lot(tmp_path, train=[_chat_row(), _chat_row()], substance={"min_train_records": 5})
    )
    assert checks["substance.train_not_empty"]["ok"] is True
    assert checks["substance.train_min_records"]["ok"] is False
    assert "required>=5 actual=2" in checks["substance.train_min_records"]["detail"]


def test_validation_may_be_empty_when_the_split_policy_asked_for_none(
    tmp_path: Path,
) -> None:
    """train_ratio 1.0 is a legitimate policy; it must not be punished."""
    checks = _checks(_lot(tmp_path, train=[_chat_row()], validation_ratio=0.0))
    assert checks["substance.validation_present_when_declared"]["ok"] is True


def test_validation_must_exist_when_the_split_policy_declared_it(tmp_path: Path) -> None:
    checks = _checks(
        _lot(tmp_path, train=[_chat_row()], validation=[], validation_ratio=0.2)
    )
    assert checks["substance.validation_present_when_declared"]["ok"] is False


def test_required_roles_catch_a_flattened_tool_trace(tmp_path: Path) -> None:
    """The persona-lot defect: 1,567 assistant messages and no tool role at all.

    Trajectories whose tool calls were flattened into assistant prose teach a
    model to narrate tool use rather than emit it. A rubric that declares the
    role must be able to say so.
    """
    flattened = [_chat_row(roles=("user", "assistant")) for _ in range(3)]
    checks = _checks(
        _lot(tmp_path, train=flattened, substance={"required_roles": ["user", "assistant", "tool"]})
    )
    assert checks["substance.required_roles_present"]["ok"] is False
    assert "missing=['tool']" in checks["substance.required_roles_present"]["detail"]


def test_required_roles_are_satisfied_when_the_role_is_really_there(
    tmp_path: Path,
) -> None:
    rows = [_chat_row(turns=6, roles=("user", "assistant", "tool"))]
    checks = _checks(
        _lot(tmp_path, train=rows, substance={"required_roles": ["user", "assistant", "tool"]})
    )
    assert checks["substance.required_roles_present"]["ok"] is True


def test_required_roles_fail_when_the_recipe_carries_no_messages(
    tmp_path: Path,
) -> None:
    """Asking for roles in a format that cannot express them is a failure."""
    checks = _checks(
        _lot(
            tmp_path,
            train=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
            substance={"required_roles": ["user"]},
        )
    )
    assert checks["substance.required_roles_present"]["ok"] is False


def test_min_turns_per_record_rejects_a_thin_record(tmp_path: Path) -> None:
    checks = _checks(
        _lot(
            tmp_path,
            train=[_chat_row(turns=6), _chat_row(turns=1)],
            substance={"min_turns_per_record": 4},
        )
    )
    assert checks["substance.min_turns_per_record"]["ok"] is False
    assert "train[1]=1" in checks["substance.min_turns_per_record"]["detail"]


def test_min_turns_per_record_passes_when_every_record_is_thick_enough(
    tmp_path: Path,
) -> None:
    checks = _checks(
        _lot(
            tmp_path,
            train=[_chat_row(turns=6), _chat_row(turns=8)],
            substance={"min_turns_per_record": 4},
        )
    )
    assert checks["substance.min_turns_per_record"]["ok"] is True


def test_turn_and_role_checks_are_absent_until_a_rubric_asks(tmp_path: Path) -> None:
    """Silence is not a pass. A check that did not run must not be reported."""
    checks = _checks(_lot(tmp_path, train=[_chat_row()]))
    assert "substance.min_turns_per_record" not in checks
    assert "substance.required_roles_present" not in checks
