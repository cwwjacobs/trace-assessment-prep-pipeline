from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .events import normalize_events, parse_event_stream, verify_event_chain
from .foundry_receipt import (
    EVALUATION_PATH_LEGACY,
    EVALUATION_PATH_THREE_TIER,
    FoundryTrustPolicy,
    FoundryReceiptError,
    load_foundry_trust_policy,
    load_receipt,
    require_receipt_by_default,
    verify_foundry_receipt,
)
from .hashing import sha256_file, sha256_json, sha256_tree
from .ingot_schema import validate_ingot
from .interactive import ARTIFACT_NAME, SCHEMA_ID, build_interactive_evidence
from .privacy import redact_text, scan_obj
from .scenario_trace import (
    ARTIFACT_NAME as SCENARIO_TRACE_ARTIFACT_NAME,
    ROUTING_ARTIFACT_NAME,
    SCHEMA_ID as SCENARIO_TRACE_SCHEMA_ID,
    build_scenario_trace_evidence,
)
from .rights_overlay import (
    RIGHTS_ASSERTED,
    RightsOverlayError,
    RightsOverlayResult,
    evaluate_rights_overlay,
    load_rights_overlay,
    unevaluated,
)
from .verify_bundle import load_json, verify_mine_bundle


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _domain_claim_binding(bundle_root: Path) -> dict[str, Any] | None:
    provenance_path = bundle_root / "scenario" / "provenance.json"
    if not provenance_path.exists():
        return None
    provenance = load_json(provenance_path)
    binding = provenance.get("domain_claim") if isinstance(provenance, dict) else None
    if binding is None:
        return None
    if not isinstance(binding, dict):
        raise ValueError("sealed scenario domain_claim binding must be an object")
    claim_id = binding.get("domain_claim_id")
    if not isinstance(claim_id, str) or not claim_id:
        raise ValueError("sealed scenario domain_claim binding lacks an id")
    if provenance.get("domain_claim_id") != claim_id:
        raise ValueError("sealed scenario domain_claim identifiers disagree")
    return dict(binding)


def _mechanical_run_status(
    run: dict[str, Any], validation: dict[str, Any]
) -> str:
    if run.get("classification") == "live_interactive":
        status = run.get("session_status")
        if not isinstance(status, str) or not status:
            raise ValueError("verified interactive run lacks session_status")
        return status
    if run.get("classification") == "live_external_process":
        status = run.get("invocation_status")
        if status not in {"COMPLETED", "FAILED", "INTERRUPTED"}:
            raise ValueError("verified external process run lacks invocation_status")
        return str(status)
    if run.get("classification") == "native_trace_import":
        status = run.get("import_status")
        if status != "COMPLETED":
            raise ValueError("verified native trace run lacks import_status")
        # Import completion is a mechanical fact, not product admission.
        return str(status)
    if run.get("classification") == "scenario_trace_import":
        status = run.get("import_status")
        if status != "COMPLETED":
            raise ValueError("verified scenario trace run lacks import_status")
        # Import completion remains distinct from Foundry/Refinery/product admission.
        return str(status)
    return str(run.get("status") or validation.get("status") or "unknown")


def _refinery_disposition(
    run: dict[str, Any],
    unresolved_privacy_findings: list[dict[str, str]],
    rights_overlay: RightsOverlayResult | None = None,
) -> tuple[str, str]:
    """Return quarantine/refinery status without weakening a stronger refusal."""
    if unresolved_privacy_findings:
        return "QUARANTINED", "QUARANTINED"
    if run.get("classification") == "native_trace_import":
        # A structurally verified historical import remains governance HOLD:
        # its run.v7 contract explicitly carries UNKNOWN rights and says it is
        # not training ready. Casting may derive a canonical ingot, but cannot
        # promote that evidence state into PASSED.
        #
        # An external rights overlay may lift that to RIGHTS_ASSERTED when it
        # binds to this exact bundle and validates. It never reaches PASSED:
        # rights evidence is not assay evidence. The sealed bundle keeps saying
        # UNKNOWN either way; that is historical truth, not a defect to patch.
        if rights_overlay is not None and rights_overlay.granted:
            return "CLEAR", RIGHTS_ASSERTED
        return "HOLD", "HOLD"
    return "CLEAR", "PASSED"


def cast_mine_bundle(
    bundle_root: Path,
    out_dir: Path,
    *,
    foundry_receipt: Path | str | dict[str, Any] | None = None,
    require_foundry_receipt: bool | None = None,
    foundry_trust_policy: FoundryTrustPolicy | Path | str | None = None,
    rights_ledger: Path | str | None = None,
) -> dict[str, Any]:
    """Verify a sealed Labyrinth bundle and cast a content-addressed ingot.

    The bundle is never written to; its tree hash is compared before and after.

    Two paths exist and every ingot records which one produced it:

    ``foundry_receipt`` supplied
        The **three-tier path**. The receipt must verify, must bind to this exact
        bundle by hash, authenticate under an explicit operator trust policy,
        use an exactly pinned evaluation pack, and carry an admissible status,
        or the cast fails closed before any derivative is written. The receipt
        hash, evaluation-pack hash and evaluation lineage are embedded in the
        ingot.

    ``foundry_receipt`` omitted
        The **legacy direct path**, kept for callers that predate Tier 2. The
        ingot's ``foundry_evaluation`` is null and its ``evaluation_path`` says
        so explicitly, so a legacy ingot can never be read as evidence that
        Foundry evaluation occurred. Set ``require_foundry_receipt=True`` (or
        ``GOLDTRACE_REQUIRE_FOUNDRY_RECEIPT=1``) to make this path an error.
    """
    bundle_root = Path(bundle_root).resolve()
    out_dir = Path(out_dir).resolve()
    try:
        out_dir.relative_to(bundle_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "out_dir must be outside the sealed bundle; refusing to write "
            f"derivatives into source evidence: {out_dir}"
        )

    if require_foundry_receipt is None:
        require_foundry_receipt = require_receipt_by_default()
    if foundry_receipt is None and require_foundry_receipt:
        raise FoundryReceiptError(
            "a Foundry evaluation receipt is required for the three-tier path but "
            "none was supplied. Run `evalfoundry evaluate-bundle` first, or pass "
            "require_foundry_receipt=False to use the legacy direct path."
        )

    # Preserve original: never write into bundle_root
    pre_hash = sha256_tree(bundle_root)
    verification = verify_mine_bundle(bundle_root)
    if not verification.ok:
        raise ValueError(f"bundle verification failed: {verification.error}")

    run = load_json(bundle_root / "run.json")
    manifest = load_json(bundle_root / "manifest.json")
    source_bundle_hash = pre_hash
    source_manifest_hash = verification.manifest_sha256 or sha256_file(
        bundle_root / "manifest.json"
    )

    # Tier-2 gate runs before output-directory creation or derivative writes.
    # Trust inputs come only from the caller's explicit policy, never from the
    # untrusted receipt itself.
    foundry_evaluation: dict[str, Any] | None = None
    if foundry_receipt is not None:
        receipt_document = (
            foundry_receipt
            if isinstance(foundry_receipt, dict)
            else load_receipt(foundry_receipt)
        )
        if isinstance(foundry_trust_policy, FoundryTrustPolicy):
            trust_policy = foundry_trust_policy
        elif foundry_trust_policy is not None:
            trust_policy = load_foundry_trust_policy(foundry_trust_policy)
        else:
            trust_policy = None
        foundry_evaluation = verify_foundry_receipt(
            receipt_document,
            source_bundle_hash=source_bundle_hash,
            source_manifest_hash=source_manifest_hash,
            labyrinth_run_id=verification.run_id,
            trust_policy=trust_policy,
        )
    evaluation_path = (
        EVALUATION_PATH_THREE_TIER
        if foundry_evaluation is not None
        else EVALUATION_PATH_LEGACY
    )

    scenario_trace_result = None
    if run.get("classification") == "scenario_trace_import":
        if foundry_evaluation is None:
            raise FoundryReceiptError(
                "scenario_trace_import requires the authenticated three-tier Foundry path"
            )
        scenario_trace_result = build_scenario_trace_evidence(
            bundle_root=bundle_root,
            run=run,
            source_bundle_hash=source_bundle_hash,
            source_manifest_hash=source_manifest_hash,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    validation = load_json(bundle_root / "validation.json") if (bundle_root / "validation.json").exists() else {}
    gaps_doc = (
        load_json(bundle_root / "evidence-gaps.json")
        if (bundle_root / "evidence-gaps.json").exists()
        else {}
    )
    cleanup = (
        load_json(bundle_root / "cleanup-receipt.json")
        if (bundle_root / "cleanup-receipt.json").exists()
        else {}
    )

    events_path = bundle_root / "events" / "events.ndjson"
    raw_events = parse_event_stream(events_path)
    chain = verify_event_chain(raw_events)
    if not chain["ok"]:
        raise ValueError(f"broken event chain: {chain['breaks']}")

    normalized = normalize_events(raw_events)
    stream_path = out_dir / "normalized_events.jsonl"
    with stream_path.open("w", encoding="utf-8") as handle:
        for item in normalized:
            handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
    stream_hash = sha256_file(stream_path)

    # Privacy scan on textual surfaces of normalized stream (sample-capped for large runs)
    findings: list[dict[str, str]] = []
    redacted_events: list[dict[str, Any]] = []
    high_severity = {"aws_access_key", "generic_api_key", "private_key_block"}
    for item in normalized:
        payload = json.dumps(item, ensure_ascii=False)
        found = scan_obj(item)
        if found:
            findings.extend(found)
            redacted_payload, _ = redact_text(payload)
            redacted_events.append(json.loads(redacted_payload))
        else:
            redacted_events.append(item)

    redacted_path = out_dir / "normalized_events.redacted.jsonl"
    with redacted_path.open("w", encoding="utf-8") as handle:
        for item in redacted_events:
            handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
    redacted_hash = sha256_file(redacted_path)
    unresolved = [f for f in findings if f.get("rule") in high_severity]

    # External rights overlay. Omitting it leaves every disposition untouched.
    # Supplying one always produces an evaluated record, including when it is
    # refused, so a requested rights check is never silently indistinguishable
    # from one that was never asked for.
    rights_overlay: RightsOverlayResult | None = None
    if rights_ledger is not None:
        try:
            overlay = load_rights_overlay(rights_ledger)
        except RightsOverlayError as exc:
            rights_overlay = unevaluated(str(exc), rights_ledger)
        else:
            rights_overlay = evaluate_rights_overlay(
                overlay,
                manifest_sha256=source_manifest_hash,
                checkpoint_head=run.get("checkpoint_head"),
                # Recomputed by the seal verifier from the sealed event chain, so
                # it stands as a second identity where checkpoint_head is null.
                event_head_sha256=(
                    verification.details.get("event_head_sha256")
                    if verification.details
                    else None
                ),
            )

    # After redaction file is written, low-severity findings are resolved derivatives.
    quarantine_status, refinery_status = _refinery_disposition(
        run, unresolved, rights_overlay
    )
    redacted_count = sum(1 for f in findings if f.get("rule") not in high_severity) if findings else 0

    # For fixture pipeline we often want PASSED mechanical units even with SEALED_WITH_GAPS.
    # Privacy findings force quarantine; gaps alone remain PASSED with explicit gap ids.

    scenario = run.get("scenario") or {}
    if isinstance(scenario, str):
        scenario_id = scenario
        scenario_version = None
    else:
        scenario_id = scenario.get("scenario_id") or scenario.get("id")
        scenario_version = scenario.get("scenario_version") or scenario.get("version")
    if scenario_trace_result is not None:
        scenario_id = scenario_trace_result.scenario_id
        scenario_version = scenario_trace_result.scenario_version
    domain_claim = _domain_claim_binding(bundle_root)
    if isinstance(scenario, dict) and scenario.get("domain_claim") != domain_claim:
        raise ValueError("run/scenario domain claim differs from sealed provenance")

    interactive_descriptor: dict[str, Any] | None = None
    if run.get("classification") == "live_interactive":
        source_index_path = bundle_root / "interactive-evidence-index.json"
        if not source_index_path.is_file():
            raise ValueError("verified interactive run lacks its evidence index")
        source_index = load_json(source_index_path)
        source_index_hash = sha256_file(source_index_path)
        run_descriptor = run.get("evidence_index")
        if (
            not isinstance(run_descriptor, dict)
            or run_descriptor.get("path") != source_index_path.name
            or run_descriptor.get("sha256") != source_index_hash
        ):
            raise ValueError("interactive run/index descriptor differs from sealed bytes")
        interactive_evidence = build_interactive_evidence(
            run=run,
            source_bundle_hash=source_bundle_hash,
            source_manifest_hash=source_manifest_hash,
            source_index=source_index,
            source_index_sha256=source_index_hash,
            redacted_events=redacted_events,
            redacted_stream_sha256=redacted_hash,
            privacy_findings_count=len(findings),
        )
        interactive_path = out_dir / ARTIFACT_NAME
        interactive_path.write_text(
            json.dumps(interactive_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        interactive_descriptor = {
            "schema_id": SCHEMA_ID,
            "path": ARTIFACT_NAME,
            "sha256": sha256_file(interactive_path),
            "source_index_sha256": source_index_hash,
            "privacy_source_sha256": redacted_hash,
        }

    scenario_trace_descriptor: dict[str, Any] | None = None
    routing_descriptor: dict[str, Any] | None = None
    if scenario_trace_result is not None:
        evidence_path = out_dir / SCENARIO_TRACE_ARTIFACT_NAME
        evidence_path.write_text(
            json.dumps(
                scenario_trace_result.evidence,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        route_path = out_dir / ROUTING_ARTIFACT_NAME
        route_path.write_text(
            json.dumps(
                scenario_trace_result.routing_receipt,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        scenario_trace_descriptor = {
            "schema_id": SCENARIO_TRACE_SCHEMA_ID,
            "path": SCENARIO_TRACE_ARTIFACT_NAME,
            "sha256": sha256_file(evidence_path),
            "source_trace_file_sha256": scenario_trace_result.evidence["source"][
                "source_trace_file_sha256"
            ],
            "source_trace_content_sha256": scenario_trace_result.evidence["source"][
                "source_trace_content_sha256"
            ],
            "verifier_content_sha256": scenario_trace_result.evidence["verifier"][
                "content_sha256"
            ],
            "trace_artifact_id": scenario_trace_result.trace_artifact_id,
            "source_family": scenario_trace_result.source_family,
            "scenario_id": scenario_trace_result.scenario_id,
            "scenario_version": scenario_trace_result.scenario_version,
        }
        routing_descriptor = {
            "schema_id": scenario_trace_result.routing_receipt["schema_id"],
            "path": ROUTING_ARTIFACT_NAME,
            "sha256": sha256_file(route_path),
            "receipt_sha256": scenario_trace_result.routing_receipt["receipt_sha256"],
            "source_family": scenario_trace_result.source_family,
            "processor": scenario_trace_result.routing_receipt["selected_route"][
                "processor"
            ],
            "output_class": scenario_trace_result.routing_receipt["selected_route"][
                "output_class"
            ],
        }

    exact_id = sha256_json(
        {
            "source_bundle_hash": source_bundle_hash,
            "event_count": len(raw_events),
            "event_head": verification.details.get("event_head_sha256") if verification.details else None,
        }
    )
    structural_id = sha256_json(
        {
            "scenario_id": scenario_id,
            "event_types": [e.get("event_type") for e in normalized[:50]],
            "event_count": len(normalized),
        }
    )

    receipt_body = {
        "stage": "refinery.cast",
        # The local source path is operational state, not portable provenance,
        # and receipts are later eligible to enter product packs.  The exact
        # bundle tree hash below is the durable source identity.
        "bundle_locator_disposition": "excluded_private_local_path",
        "source_bundle_hash_before": pre_hash,
        "evaluation_path": evaluation_path,
        "foundry_evaluation": foundry_evaluation,
        "verification": verification.details,
        "chain": chain,
        "transformations": [
            "verify_mine_bundle",
            "parse_events",
            "verify_event_chain",
            "normalize_events",
            "privacy_scan",
            *( ["derive_privacy_treated_interactive_evidence"] if interactive_descriptor else [] ),
            *(
                [
                    "derive_privacy_bounded_scenario_trace_evidence",
                    "issue_processor_routing_receipt",
                ]
                if scenario_trace_descriptor is not None
                else []
            ),
            "cast_ingot",
            "preserve_domain_claim_binding",
            *(["evaluate_rights_overlay"] if rights_overlay is not None else []),
        ],
        "privacy_findings_count": len(findings),
        "rights_overlay": (
            rights_overlay.to_record() if rights_overlay is not None else {"requested": False}
        ),
        "created_at": _utc_now(),
    }
    # provisional receipt hash without self-reference
    receipt_hash = sha256_json(receipt_body)
    receipt_body["receipt_hash"] = receipt_hash

    ingot_id = f"GTI-{source_bundle_hash[:12]}"
    ingot: dict[str, Any] = {
        "schema_id": "goldtrace.refinery.ingot.v1",
        "ingot_id": ingot_id,
        "source_class": "goldtrace_mine_bundle",
        "source_bundle_hash": source_bundle_hash,
        "source_manifest_hash": source_manifest_hash,
        "evaluation_path": evaluation_path,
        "foundry_evaluation": foundry_evaluation,
        "mine_run_id": verification.run_id or run.get("run_id"),
        "domain_claim_id": (
            domain_claim.get("domain_claim_id") if domain_claim is not None else None
        ),
        "domain_claim": domain_claim,
        "plugin_id": scenario_id,
        "campaign_id": None,
        "task_family_id": scenario_id,
        "scenario_id": scenario_id,
        "scenario_version": scenario_version,
        "configuration_id": None,
        "mechanical_run_status": _mechanical_run_status(run, validation),
        "seal_status": verification.seal_status or manifest.get("seal_status") or "unknown",
        "evidence_gap_ids": list(verification.evidence_gap_ids or []),
        "normalized_event_stream_ref": "normalized_events.jsonl",
        "normalized_event_stream_hash": stream_hash,
        "artifact_manifest_ref": "manifest.json" if (bundle_root / "manifest.json").exists() else None,
        "artifact_manifest_hash": source_manifest_hash,
        "deterministic_results": {
            "bundle_verification": "PASS",
            "event_chain": "PASS" if chain["ok"] else "FAIL",
            "validation": validation,
            "cleanup_present": bool(cleanup),
            "evidence_gaps": gaps_doc if isinstance(gaps_doc, dict) else {"raw": gaps_doc},
            "event_count": len(raw_events),
            **(
                {"interactive_evidence": interactive_descriptor}
                if interactive_descriptor is not None
                else {}
            ),
            **(
                {
                    "scenario_trace_evidence": scenario_trace_descriptor,
                    "processor_routing": routing_descriptor,
                }
                if scenario_trace_descriptor is not None
                and routing_descriptor is not None
                else {}
            ),
        },
        "redaction": {
            "findings_count": len(findings),
            "redacted_records": redacted_count,
            "findings_by_rule": {
                rule: sum(1 for finding in findings if finding.get("rule") == rule)
                for rule in sorted({str(finding.get("rule")) for finding in findings})
            },
            "unresolved_high_severity": len(unresolved),
            "before_hash": stream_hash,
            "after_hash": redacted_hash,
            "redacted_stream_ref": "normalized_events.redacted.jsonl",
        },
        "quarantine": {
            # Privacy findings only. A non-binding or invalid rights overlay is
            # not a defect in the sealed evidence, so it never lands here; it is
            # reported under `rights_overlay` and in its own receipt instead.
            "status": quarantine_status,
            "reasons": [f["rule"] for f in findings[:20]],
        },
        "rights_overlay": (
            rights_overlay.to_record() if rights_overlay is not None else {"requested": False}
        ),
        "dedup": {"exact_id": exact_id, "structural_id": structural_id},
        "lineage_parents": [
            {
                "kind": "mine_bundle",
                "run_id": verification.run_id,
                "bundle_hash": source_bundle_hash,
                "manifest_hash": source_manifest_hash,
            },
            *(
                [
                    {
                        "kind": "scenario_trace",
                        "trace_artifact_id": scenario_trace_result.trace_artifact_id,
                        "source_family": scenario_trace_result.source_family,
                        "evidence_sha256": scenario_trace_result.evidence[
                            "evidence_sha256"
                        ],
                        "routing_receipt_sha256": scenario_trace_result.routing_receipt[
                            "receipt_sha256"
                        ],
                    }
                ]
                if scenario_trace_result is not None
                else []
            ),
            *(
                [
                    {
                        "kind": "foundry_evaluation",
                        "receipt_hash": foundry_evaluation["receipt_hash"],
                        "receipt_schema_id": foundry_evaluation["receipt_schema_id"],
                        "status": foundry_evaluation["status"],
                        "evaluation_pack_id": foundry_evaluation["evaluation_pack_id"],
                        "evaluation_pack_version": foundry_evaluation["evaluation_pack_version"],
                        "evaluation_pack_hash": foundry_evaluation["evaluation_pack_hash"],
                        "foundry_code_sha256": foundry_evaluation["foundry_code_sha256"],
                    }
                ]
                if foundry_evaluation is not None
                else []
            ),
            *(
                [
                    {
                        "kind": "domain_claim",
                        "domain_claim_id": domain_claim["domain_claim_id"],
                        "claim_sha256": domain_claim["claim_sha256"],
                        "status": domain_claim["status"],
                        "readiness_verdict": domain_claim["readiness_verdict"],
                        "approval_state": domain_claim["approval_state"],
                    }
                ]
                if domain_claim is not None
                else []
            ),
        ],
        "refinery_receipt_hash": receipt_hash,
        "refinery_status": refinery_status,
        "trajectory_summary": {
            "event_count": len(raw_events),
            "normalized_count": len(normalized),
            "seal_status": verification.seal_status,
            **(
                {
                    "source_family": scenario_trace_result.source_family,
                    "trace_artifact_id": scenario_trace_result.trace_artifact_id,
                    "scenario_id": scenario_trace_result.scenario_id,
                    "failure_and_correction_count": scenario_trace_result.evidence[
                        "trajectory"
                    ]["failure_and_correction_count"],
                    "source_step_count": len(
                        scenario_trace_result.evidence["trajectory"]["steps"]
                    ),
                    "independent_verifier_status": scenario_trace_result.evidence[
                        "verifier"
                    ]["status"],
                }
                if scenario_trace_result is not None
                else {}
            ),
        },
        "created_at": _utc_now(),
    }

    errors = validate_ingot(ingot)
    if errors:
        raise ValueError(f"ingot schema invalid: {errors}")

    ingot_path = out_dir / "ingot.json"
    receipt_path = out_dir / "refinery-receipt.json"
    ingot_path.write_text(json.dumps(ingot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt_body, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # A requested overlay always leaves its own receipt on disk, granted or not,
    # so an operator can see that the check ran and why it landed where it did.
    rights_overlay_receipt_path: Path | None = None
    if rights_overlay is not None:
        rights_overlay_receipt_path = out_dir / "rights-overlay-receipt.json"
        rights_overlay_receipt_path.write_text(
            json.dumps(
                {
                    "stage": "refinery.rights_overlay",
                    "ingot_id": ingot_id,
                    "source_bundle_hash": source_bundle_hash,
                    "source_manifest_hash": source_manifest_hash,
                    "resulting_refinery_status": refinery_status,
                    **rights_overlay.to_record(),
                    "created_at": _utc_now(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    post_hash = sha256_tree(bundle_root)
    if post_hash != pre_hash:
        raise RuntimeError("bundle was modified during cast — abort")

    result = {
        "ingot_path": str(ingot_path),
        "receipt_path": str(receipt_path),
        "ingot_id": ingot_id,
        "ingot_hash": sha256_file(ingot_path),
        "source_bundle_hash": source_bundle_hash,
        "source_manifest_hash": source_manifest_hash,
        "refinery_status": refinery_status,
        "evaluation_path": evaluation_path,
        "foundry_receipt_hash": (
            foundry_evaluation["receipt_hash"] if foundry_evaluation is not None else None
        ),
        "evaluation_pack_hash": (
            foundry_evaluation["evaluation_pack_hash"] if foundry_evaluation is not None else None
        ),
        "bundle_unmodified": True,
        "rights_overlay_requested": rights_overlay is not None,
        "rights_overlay_evaluated": (
            rights_overlay.evaluated if rights_overlay is not None else False
        ),
        "rights_overlay_granted": (
            rights_overlay.granted if rights_overlay is not None else False
        ),
        "rights_overlay_reason": (
            rights_overlay.reason if rights_overlay is not None else None
        ),
        "rights_overlay_receipt_path": (
            str(rights_overlay_receipt_path) if rights_overlay_receipt_path else None
        ),
    }
    if interactive_descriptor is not None:
        result["interactive_evidence_path"] = str(out_dir / ARTIFACT_NAME)
        result["interactive_evidence_hash"] = interactive_descriptor["sha256"]
    if scenario_trace_descriptor is not None and routing_descriptor is not None:
        result["scenario_trace_evidence_path"] = str(
            out_dir / SCENARIO_TRACE_ARTIFACT_NAME
        )
        result["scenario_trace_evidence_hash"] = scenario_trace_descriptor["sha256"]
        result["processor_routing_receipt_path"] = str(out_dir / ROUTING_ARTIFACT_NAME)
        result["processor_routing_receipt_hash"] = routing_descriptor["sha256"]
        result["source_family"] = routing_descriptor["source_family"]
        result["processor"] = routing_descriptor["processor"]
        result["output_class"] = routing_descriptor["output_class"]
    return result
