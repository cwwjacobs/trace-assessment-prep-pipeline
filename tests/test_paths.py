from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]


def test_explicit_labyrinth_source_outranks_an_installed_package(tmp_path: Path) -> None:
    selected = tmp_path / "selected"
    stale = tmp_path / "stale"
    for root, marker in ((selected, "selected"), (stale, "stale")):
        package = root / "goldentrace"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            f"ORIGIN = {marker!r}\n", encoding="utf-8"
        )

    environment = os.environ.copy()
    environment["GOLDENTRACE_SRC"] = str(selected)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(COMPONENT_ROOT / "src"), str(stale))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from goldtrace_refinery.paths import ensure_labyrinth_importable; "
                "chosen=ensure_labyrinth_importable(); import goldentrace; "
                "print(chosen); print(goldentrace.ORIGIN)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [str(selected), "selected"]
