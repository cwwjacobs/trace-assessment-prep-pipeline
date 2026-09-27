"""Withdrawal is how the Vault says something is no longer held.

It had no tests, and it did not work. ``withdraw`` wrote ``reason`` into a
record whose contract sets ``additionalProperties: false`` and never defines
it, left three fields null where the contract required strings, asserted
``hallmark_status: PASS`` on the very row retracting a pass, and — unlike
``admit`` — never validated what it wrote. Every withdrawal produced a row
that violated its own contract, unnoticed, because nothing read it back.

The contract now describes both actions in one record type. An admission binds
a lot to the Hallmark report that passed it; a withdrawal carries
``action: withdraw`` and states why. The admission row is never removed: a
withdrawal that deleted it would erase the claim it exists to correct.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from goldtrace_refinery.paths import find_contract
from goldtrace_vault.store import LocalVault, VaultError

DIGEST = "b" * 64


def _entry_validator() -> Draft202012Validator:
    return Draft202012Validator(
        json.loads(find_contract("vault-entry.v1.schema.json").read_text(encoding="utf-8"))
    )


def _vault(tmp_path: Path) -> LocalVault:
    return LocalVault(tmp_path / "vault", production_mode=False)


def _ledger_rows(vault: LocalVault) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in vault.ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_a_withdrawal_validates_against_the_entry_contract(tmp_path: Path) -> None:
    """The defect this file exists for: withdrawals violated their own contract."""
    entry = _vault(tmp_path).withdraw("product.x", "1.0.0", "admitted without verification")
    errors = [error.message for error in _entry_validator().iter_errors(entry)]
    assert not errors, errors


def test_a_withdrawal_does_not_claim_the_product_passed(tmp_path: Path) -> None:
    """A withdrawn row asserting PASS leaves a false claim standing."""
    entry = _vault(tmp_path).withdraw("product.x", "1.0.0", "rights never cleared")
    assert entry["hallmark_status"] == "WITHDRAWN"
    assert entry["sale_eligibility"] == "withdrawn"
    assert entry["rights_status"] == "withdrawn"


@pytest.mark.parametrize("reason", ["", "   ", None, 7])
def test_a_withdrawal_must_state_a_reason(tmp_path: Path, reason: Any) -> None:
    with pytest.raises(VaultError, match="reason"):
        _vault(tmp_path).withdraw("product.x", "1.0.0", reason)


def test_a_withdrawal_binds_the_admission_it_retracts(tmp_path: Path) -> None:
    entry = _vault(tmp_path).withdraw(
        "product.x",
        "1.0.0",
        "admission bypassed verification",
        retracted_admission_receipt_hash=DIGEST,
    )
    assert entry["retracted_admission_receipt_hash"] == DIGEST
    assert not list(_entry_validator().iter_errors(entry))


def test_a_withdrawal_refuses_a_binding_that_is_not_a_digest(tmp_path: Path) -> None:
    with pytest.raises(VaultError, match="sha256"):
        _vault(tmp_path).withdraw(
            "product.x", "1.0.0", "reason", retracted_admission_receipt_hash="not-a-digest"
        )


def test_the_ledger_is_append_only(tmp_path: Path) -> None:
    """A withdrawal must not erase the claim it is correcting."""
    vault = _vault(tmp_path)
    first = vault.withdraw("product.x", "1.0.0", "first correction")
    assert len(_ledger_rows(vault)) == 1
    vault.withdraw("product.y", "2.0.0", "second correction")
    rows = _ledger_rows(vault)
    assert len(rows) == 2
    assert rows[0] == first, "the earlier row was rewritten"


def test_the_receipt_hash_covers_the_reason(tmp_path: Path) -> None:
    """Otherwise the stated reason could be changed without breaking the hash."""
    vault = _vault(tmp_path)
    one = vault.withdraw("product.x", "1.0.0", "reason one")
    two = vault.withdraw("product.x", "1.0.0", "reason two")
    assert one["admission_receipt_hash"] != two["admission_receipt_hash"]


def test_an_admission_may_not_carry_a_withdrawal_reason() -> None:
    """An admission that needs a reason is not an admission."""
    admission = {
        "schema_id": "goldtrace.vault.entry.v1",
        "product_id": "product.x",
        "product_version": "1.0.0",
        "product_lot_hash": DIGEST,
        "hallmark_report_hash": DIGEST,
        "hallmark_status": "PASS",
        "admitted_at": "2026-08-12T00:00:00Z",
        "storage_location": "cas/bb/" + DIGEST,
        "rights_status": "held",
        "sale_eligibility": "held",
        "admission_receipt_hash": DIGEST,
        "action": "admit",
    }
    assert not list(_entry_validator().iter_errors(admission))
    assert list(_entry_validator().iter_errors({**admission, "reason": "why"}))


def test_an_admission_may_not_leave_its_bindings_null() -> None:
    """Null lot and report hashes are only meaningful on a withdrawal."""
    admission = {
        "schema_id": "goldtrace.vault.entry.v1",
        "product_id": "product.x",
        "product_version": "1.0.0",
        "product_lot_hash": None,
        "hallmark_report_hash": DIGEST,
        "hallmark_status": "PASS",
        "admitted_at": "2026-08-12T00:00:00Z",
        "storage_location": "cas/bb/" + DIGEST,
        "rights_status": "held",
        "sale_eligibility": "held",
        "admission_receipt_hash": DIGEST,
        "action": "admit",
    }
    assert list(_entry_validator().iter_errors(admission))


def test_a_legacy_row_without_an_action_is_read_as_an_admission() -> None:
    """Rows predating the action field must keep validating as admissions."""
    legacy = {
        "schema_id": "goldtrace.vault.entry.v1",
        "product_id": "product.x",
        "product_version": "1.0.0",
        "product_lot_hash": DIGEST,
        "hallmark_report_hash": DIGEST,
        "hallmark_status": "PASS",
        "admitted_at": "2026-08-05T05:16:05Z",
        "storage_location": "cas/bb/" + DIGEST,
        "rights_status": "held",
        "sale_eligibility": "held",
        "admission_receipt_hash": DIGEST,
    }
    assert not list(_entry_validator().iter_errors(legacy))
    assert list(_entry_validator().iter_errors({**legacy, "hallmark_status": "WITHDRAWN"}))


def test_the_vault_requires_the_unconditional_substance_checks() -> None:
    """Admission must not accept a report that never asked whether anything is there.

    Every other required check is satisfied by an empty product. These two are
    the reason a lot of zero bytes can no longer reach the vault.
    """
    from goldtrace_vault.store import _REQUIRED_HALLMARK_PASS_CHECKS

    assert "substance.train_not_empty" in _REQUIRED_HALLMARK_PASS_CHECKS
    assert "substance.train_min_records" in _REQUIRED_HALLMARK_PASS_CHECKS


def test_the_vault_does_not_require_the_conditional_substance_checks() -> None:
    """Hallmark emits these only when a rubric asks; demanding them would
    reject every lot whose rubric stayed silent. A check that did not run
    must not be required."""
    from goldtrace_vault.store import _REQUIRED_HALLMARK_PASS_CHECKS

    assert "substance.min_turns_per_record" not in _REQUIRED_HALLMARK_PASS_CHECKS
    assert "substance.required_roles_present" not in _REQUIRED_HALLMARK_PASS_CHECKS
