# SentinelSanctuary cyber Ingot mold

## Claim boundary

This is a deterministic, declarative **post-Refinery mold**. It does not run a
model, execute mold-provided code, sign a receipt, invoke Hallmark, submit to
Vault, or grant sale eligibility. It accepts only an existing three-tier
Labyrinth → Foundry → Refinery Ingot whose Foundry decision is authenticated,
exact-pack bound, deterministic, and `PASS`, and whose Refinery state is
`PASSED` and quarantine state is `CLEAR`.

Every cast remains `HOLD`. A defensive scenario may legitimately carry a
Cinderfield `DENY`; Foundry's bound oracle, not the spelling of the execution
gate, determines whether the scenario passes evaluation.

## Deterministic products

The cast produces two canonical JSON files:

- `ingot-core.json` is the byte-stable semantic core. Its identity includes
  stable source, mold, domain-record, oracle, split, OpenAI witness, and
  Cinderfield witness facts. It deliberately excludes wall-clock timestamps
  and upstream receipt hashes whose authenticated wrappers may be reissued.
- `cast-request.json` binds the exact source-Ingot, record, mold, Foundry
  receipt, and Refinery receipt identities. It is explicitly `UNSIGNED`,
  `NOT_RUN` by Hallmark, `NOT_SUBMITTED` to Vault, and `HOLD` for sale.

Thus replay has three distinct meanings:

1. `CAPTURED_EXACT` re-reduces recorded request, response, and event bytes
   without another provider call.
2. `DETERMINISTIC_POLICY` replays pinned reducers, oracle, and policy over the
   same evidence.
3. `FRESH_SEMANTIC_NONDETERMINISTIC` makes a new provider observation and
   compares separately content-addressed results. It never claims byte-identical
   model output.

## Cyber record shape

The separately hashed cyber schema permits only bounded metadata and hashes.
It binds:

- operator authorization, commercial-rights, privacy, retention, provider
  disclosure, and raw-evidence disposition policies;
- oracle, scoring rules, split policy and assignment, contamination scan,
  evaluation cases/results, and one explicit train/validation/test/holdout role;
- six leak-resistant grouping namespaces: source, family, fixture, mechanic,
  narrative, and checkpoint. All reruns and derivatives retain their parent
  groups;
- exact OpenAI API family, endpoint hash, requested/returned model IDs, snapshot
  status, request/response/parameter/tool/usage/request-ID hashes, data-control
  profile, storage request, seed support, and the explicit statement that output
  reproducibility is not claimed;
- Cinderfield envelope, adapter, observer, Road, runtime assets, job image,
  preflight, proxy peer, egress, encrypted evidence, lifecycle, cleanup, and
  credential-channel witnesses.

Raw target bytes, prompts, responses, credentials, and evidence content are not
fields in this shape. The default mold refuses provider application-state
storage (`store_requested` must be false); a policy hash alone never launders
the fact that authorized data was disclosed to the API provider.

## Invocation

From the Refinery component:

```bash
PYTHONPATH=src python3 -m goldtrace_refinery.ingot_mold \
  --ingot /path/to/refinery/ingot.json \
  --record /path/to/sentinel-cyber-record.json \
  --out /new/or/byte-identical/cast-directory
```

For an installed wheel, omit `PYTHONPATH=src`.

Inputs are bounded regular files read through one `O_NOFOLLOW` descriptor.
Outputs are private, fsynced, non-overwriting, and idempotent only when every
existing byte matches. The module never uses the network.

## Contracts

- `contracts/ingot-mold-module.v1.schema.json` — executable engine's declarative
  module boundary; module values are data, never authority-bearing code.
- `contracts/sentinel-sanctuary.cyber.v1.json` — default policy module.
- `contracts/sentinel-sanctuary.cyber-record.v1.schema.json` — cyber payload.
- `contracts/domain-ingot-core.v1.schema.json` — reusable deterministic envelope.
- `contracts/domain-ingot-cast-request.v1.schema.json` — exact unsigned handoff.

The request enumerates every remaining gate. At minimum those are authenticated
issuance, asymmetric Foundry attestation, authenticated Hallmark, Vault-owned
recomputation, and human sale release. Weak assurance, structural-only
ciphertext validation, unresolved gaps, rights or contamination holds, local
HMAC observer trust, unverified cleanup, an open credential channel, an
unpinned model, or an incomplete execution adds a named gate rather than being
silently normalized away.

## Current readiness

This mold is usable for deterministic local casting and refusal tests. It is
not evidence that the Cinderfield → Labyrinth adapter exists, that supported-host
Firecracker execution passed, that evidence can be decrypted, that OpenAI was
called, that Hallmark review is authenticated, or that Vault independently
recomputed and admitted a lot. Those remain external closure conditions.
