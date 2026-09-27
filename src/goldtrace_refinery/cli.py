from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cast import cast_mine_bundle
from .foundry_receipt import FoundryReceiptError
from .verify_bundle import verify_mine_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="goldtrace-refinery",
        description=(
            "Tier-3 verification and ingot casting. Casting with --foundry-receipt "
            "runs the sequential three-tier path; casting without one runs the "
            "legacy direct path, which is recorded in the ingot as such."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify", help="Verify a sealed Labyrinth bundle")
    p_verify.add_argument("bundle")

    p_cast = sub.add_parser("cast", help="Cast an ingot from a sealed Labyrinth bundle")
    p_cast.add_argument("bundle")
    p_cast.add_argument("--out", required=True)
    p_cast.add_argument(
        "--foundry-receipt",
        default=None,
        help=(
            "Path to a Foundry evaluation receipt. Supplying one selects the "
            "three-tier path: the receipt must bind to this bundle by hash and "
            "carry an admissible status, or the cast fails closed."
        ),
    )
    p_cast.add_argument(
        "--require-foundry-receipt",
        action="store_true",
        help=(
            "Refuse to cast without a Foundry receipt. Equivalent to setting "
            "GOLDTRACE_REQUIRE_FOUNDRY_RECEIPT=1."
        ),
    )
    p_cast.add_argument(
        "--foundry-trust-policy",
        default=None,
        help=(
            "Operator-controlled JSON policy naming the trusted Foundry issuer/key "
            "and exact admitted evaluation-pack identities/hashes. Required when "
            "--foundry-receipt is supplied."
        ),
    )
    p_cast.add_argument(
        "--rights-ledger",
        default=None,
        help=(
            "External append-only rights ledger asserting operator rights over "
            "this bundle. Granting requires two independently recomputed non-null "
            "bindings from {manifest_sha256, checkpoint_head, event_head_sha256}; "
            "null equality is never a binding. Every binding entry must carry an "
            "admissible rights_status with a valid assertion. On success the "
            "ingot is RIGHTS_ASSERTED, never PASSED: rights evidence is not "
            "assay evidence. Any failure keeps the ingot at HOLD and is reported "
            "in rights-overlay-receipt.json. Omitting this changes nothing."
        ),
    )
    args = parser.parse_args(argv)

    if args.cmd == "verify":
        result = verify_mine_bundle(Path(args.bundle))
        print(json.dumps(result.__dict__, indent=2, default=str))
        return 0 if result.ok else 1

    if args.cmd == "cast":
        try:
            out = cast_mine_bundle(
                Path(args.bundle),
                Path(args.out),
                foundry_receipt=Path(args.foundry_receipt) if args.foundry_receipt else None,
                require_foundry_receipt=True if args.require_foundry_receipt else None,
                foundry_trust_policy=(
                    Path(args.foundry_trust_policy)
                    if args.foundry_trust_policy
                    else None
                ),
                rights_ledger=Path(args.rights_ledger) if args.rights_ledger else None,
            )
        except FoundryReceiptError as exc:
            # Distinct exit code so a caller can tell a Tier-2 gate refusal from
            # a malformed bundle.
            print(json.dumps({"ok": False, "error": str(exc), "reason": "foundry_gate"}), file=sys.stderr)
            return 3
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
            return 1
        print(json.dumps({"ok": True, **out}, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
