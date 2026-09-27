"""Worked example: make one synthetic agent trace safe to assess.

Everything in messy_trace.jsonl is invented for this demo: the person, the
email, the phone number, the key and the host do not exist.

    python3 examples/demo.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from goldtrace_mint.factory.contamination import ContaminationChecker  # noqa: E402
from goldtrace_mint.factory.sanitizer import PIISanitizer, sanitize_file  # noqa: E402

RAW = HERE / "messy_trace.jsonl"
OUT = HERE / "output"
PACE = float(os.environ.get("DEMO_PACE", "0"))  # seconds between steps, for recordings


def rule(title: str) -> None:
    time.sleep(PACE)
    print(f"\n== {title} " + "=" * max(0, 60 - len(title)), flush=True)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    turns = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]

    rule("1. Raw trace (synthetic)")
    for t in turns[:3]:
        print(f"[{t['role']}] {t['content']}")

    rule("2. Privacy scrub")
    receipt = sanitize_file(
        RAW,
        OUT / "clean_trace.jsonl",
        receipt_path=OUT / "clean_trace.receipt.json",
        sanitizer=PIISanitizer(workspace_home="/home/operator"),
    )
    clean = [json.loads(line) for line in (OUT / "clean_trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for t in clean[:3]:
        print(f"[{t['role']}] {t['content']}")
    print()
    for category, count in sorted(receipt.findings_by_category.items()):
        print(f"  redacted {category:<14} x{count}")
    print(f"  residual findings: {'none' if receipt.is_clean else 'PRESENT'}")
    print(f"  output sha256:  {receipt.output_hash[:16]}...")
    print(f"  receipt sha256: {receipt.receipt_hash[:16]}...  -> output/clean_trace.receipt.json")

    rule("3. Benchmark contamination check")
    checker = ContaminationChecker()
    flagged = []
    unchecked = 0
    for t in clean:
        report = checker.check_turn(t, turn_index=t["turn"])
        status = "LEAK" if report.benchmark_matches else ("ok" if report.total_ngrams else "too short to check")
        print(f"  turn {t['turn']} [{t['role']:<9}] {status}")
        if report.benchmark_matches:
            flagged.append((t["turn"], report))
        elif not report.total_ngrams:
            unchecked += 1
    for turn, report in flagged:
        ids = sorted({s.benchmark_id for s in report.flagged_spans})
        print(f"\n  turn {turn} overlaps {', '.join(ids)} "
              f"({report.contaminated_ngrams}/{report.total_ngrams} 13-grams)")
        print("  -> quarantined: it must not reach training or eval data")
    (OUT / "contamination.json").write_text(
        json.dumps({str(turn): r.to_dict() for turn, r in flagged}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rule("Result")
    ready = len(clean) - len(flagged)
    print(f"  {ready} of {len(clean)} turns scrubbed and ready for assessment "
          f"({unchecked} too short for a 13-gram check)")
    print(f"  {len(flagged)} quarantined for benchmark overlap")
    print("  every step leaves a receipt in examples/output/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
