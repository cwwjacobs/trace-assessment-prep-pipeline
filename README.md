# trace-assessment-prep-pipeline

**Turns raw AI-agent traces into clean, verified, documented data, ready for
assessment and, if the assessment calls for it, fine-tuning.**

Part of the [Gold Trace Dataworks](https://gtdataworks.com/) pipeline (internally
"GTDataworks Refinery"). An agent failure can call for anything from "no action
needed" to a prompt, tool or guardrail fix to a fine-tune. Making that call
requires evidence you can trust. This pipeline produces it: every step leaves a
receipt, and nothing is silently repaired or guessed.

- **Privacy scrubbing:** emails, phone numbers, IPs, API keys (OpenAI, Anthropic,
  AWS, GitHub), private keys, JWTs, SSNs, card numbers and home-directory paths
  are replaced with fixed tokens such as `[REDACTED_EMAIL]`. Every run writes a
  SHA-256 receipt, and any leftover finding fails the release.
- **Benchmark contamination checks:** 13-gram overlap and MinHash similarity
  against benchmark items, so eval questions do not leak into training data.
- **License gate:** accepts permissive licenses only (MIT, Apache-2.0, BSD,
  ISC, 0BSD, Unlicense, CC0); rejects GPL-family, MPL, EUPL, SSPL, proprietary
  and unlicensed sources. Fails closed.
- **Dataset documentation:** generates a `DATACARD.md` in the style of
  *Datasheets for Datasets* (Gebru et al.), with token statistics and provenance.
- **Training exports:** loss-masked JSONL (OpenAI, ShareGPT and Anthropic tool
  formats) and typed Parquet.
- **Provenance end to end:** every product ("ingot") embeds the hashes of the run
  it came from and the evaluation receipt that admitted it. Hallmark verifies
  that output files match their claims; Vault stores admitted product lots by
  content hash.

A check that cannot pass stops the release and says why.

## What it does and does not claim

- The contamination checker ships a small **sample** signature set: 35 items
  drawn from HumanEval, MBPP, SWE-bench and GSM8K. It demonstrates the method;
  it is not full benchmark coverage. For real decontamination, pass the complete
  benchmark sets through `ContaminationChecker(benchmark_signatures=...)`.
- The sensitive-phrase registry starts **empty**. Aliases, codenames and other
  corpus-specific phrases are registered per deployment; none ship with the code.
- Proof-of-lift reports (`ASSAY_REPORT.md`) use an exact two-sided McNemar test
  on paired pass/fail results, with a 95% confidence interval and an explicit
  significant / not-significant flag at α = 0.05.
- For inputs shorter than 13 tokens, the contamination report has no n-grams to
  compare and currently prints "CERTIFIED CLEAN". Treat that as "not checked".

## Layout

```
contracts/                    Versioned JSON Schemas (Draft 2020-12)
src/goldtrace_refinery/       Tier-3 CLI, bundle verification, ingot casting,
                              release privacy, rights and pack records
src/goldtrace_mint/factory/   Library modules:
    sanitizer.py                PII / secret scrubbing and license verification
    contamination.py            13-gram + MinHash benchmark overlap checks
    deduplicator.py             exact and near-duplicate pruning
    datacard.py                 DATACARD.md generator
    packager.py                 JSONL / Parquet / Arrow export
    loss_masking.py             per-message training loss weights
    assay_report.py             proof-of-lift statistics
src/goldtrace_hallmark/       Verifies that minted files match their claims
src/goldtrace_vault/          Content-addressed store for admitted product lots
tests/                        509 tests
```

## Quick start

Requires Python 3.10+.

```bash
pip install .              # base install (jsonschema is the only runtime dependency)
pip install '.[all,dev]'   # adds Parquet/Arrow, Zstandard and pytest

pytest -q
```

A base install without the optional backends still passes; the Parquet and
Zstandard tests are skipped. `tests/test_native_products.py` needs the Labyrinth
(Tier 1) `goldentrace` package, which is not in this repository, and is skipped
when it is absent.

The installed CLI:

```bash
goldtrace-refinery verify <sealed-bundle-dir>
goldtrace-refinery cast   <sealed-bundle-dir> --out <out-dir>
```

Standalone file sanitizer:

```bash
python3 -m goldtrace_mint.factory.sanitizer input.jsonl output.jsonl \
  --extra-phrase "Project Nightjar"
```

### Authenticated three-tier casting

The unkeyed `receipt_hash` detects accidental changes but cannot establish who
made a Foundry decision: anyone can edit a HOLD to PASS and recompute that hash.
For a three-tier cast, Refinery therefore requires both an HMAC-authenticated
Foundry receipt and a separate operator-controlled trust policy. No key is
shipped, generated, or inferred from a receipt.

Provision the same private key to the Foundry issuer and this Refinery trust
boundary outside the repository:

```bash
umask 077
openssl rand -out /operator/secrets/foundry-receipt.key 32
chmod 600 /operator/secrets/foundry-receipt.key
```

The key must be a regular, non-symlink file containing 32–4096 bytes and, on
POSIX systems, must be owned by the current user and inaccessible to group or
other users. The trust-policy file must likewise be regular, non-symlink and
current-user-owned, and it must not be group/other-writable. Create a policy such
as `/operator/config/foundry-trust.json`; its pack tuple must match the
independently selected pack file exactly. This example pins the currently
shipped `labyrinth-bundle-admission-v1/pack.json` bytes:

```json
{
  "schema_id": "goldtrace.refinery.foundry_trust_policy.v1",
  "issuer_id": "foundry.production",
  "authentication_method": "hmac-sha256",
  "key_id": "foundry-key-2026-08",
  "authentication_key_file": "/operator/secrets/foundry-receipt.key",
  "evaluation_packs": [
    {
      "pack_id": "labyrinth-bundle-admission",
      "pack_version": "1.0.0",
      "pack_hash": "af95545fcad2ac85beb2e1b3f71a99937b254e2348de53afc13e39cd9cbe8793"
    }
  ]
}
```

Verify a different pack pin from the exact `pack.json` selected by the operator
(for example with `sha256sum`) before changing that tuple. Do not copy identity
or hash defaults out of an untrusted receipt.

```bash
goldtrace-refinery cast /path/to/sealed/run-bundle \
  --out /path/to/cast-output \
  --foundry-receipt /path/to/foundry-receipt.json \
  --foundry-trust-policy /operator/config/foundry-trust.json \
  --require-foundry-receipt
```

Unsigned receipts—including Foundry output explicitly created with
`--allow-unsigned-receipt`—are always rejected on this path. Issuer/key mismatch,
an unpinned pack id/version/hash, inconsistent signed status/check/count data,
and HOLD/FAIL/ERROR all fail before normalized or redacted derivatives are
written. PASS is an invariant of the production three-tier gate; neither the
CLI nor the library exposes a status-widening override.

### Library APIs

Contamination scanning and packaging are Python library APIs, not CLIs:

```python
from goldtrace_mint.factory.contamination import ContaminationChecker
from goldtrace_mint.factory.packager import HAS_PYARROW, HAS_ZSTD

checker = ContaminationChecker()          # sample signatures; pass your own sets
report = checker.check_text("...")
print(report.summary())
```

## License

Copyright (c) 2026 Gold Trace Dataworks. All rights reserved. The source is
published for review; see `LICENSE` for terms. For dataset or licensing work,
visit [gtdataworks.com](https://gtdataworks.com/).
