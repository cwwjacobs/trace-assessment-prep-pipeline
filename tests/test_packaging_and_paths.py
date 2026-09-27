"""Packaging and path-portability contracts for the Refinery component.

These tests fail on any regression that would make the component work only on
one operator's machine, or make an installed wheel unable to find its own
contracts.
"""

from __future__ import annotations

import ast
import configparser
import re
import sys
from pathlib import Path

import pytest

COMPONENT_ROOT = Path(__file__).resolve().parents[1]

#: Absolute-path shapes that must never appear in shipped code or config.
FORBIDDEN_PATH_PATTERN = re.compile(r"(?<![\w.])/(?:home|Users)/[A-Za-z0-9._-]+/")

#: Files whose content is operator-facing configuration or shipped source.
SCANNED_SUFFIXES = {".py", ".ini", ".toml", ".cfg", ".json"}


#: Calls whose string arguments become real filesystem targets.
_FILESYSTEM_CALLS = {"Path", "PosixPath", "open", "chdir", "listdir", "walk", "glob", "iglob"}


def _scanned_files() -> list[Path]:
    roots = [
        COMPONENT_ROOT / "src",
        COMPONENT_ROOT / "donor_components",
        COMPONENT_ROOT / "tests",
    ]
    files = [
        COMPONENT_ROOT / "pytest.ini",
        COMPONENT_ROOT / "pyproject.toml",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in SCANNED_SUFFIXES and "__pycache__" not in path.parts:
                files.append(path)
    return [f for f in files if f.is_file()]


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _filesystem_path_literals(source: str) -> list[str]:
    """Absolute operator paths that are used as filesystem targets.

    Deliberately narrower than a text search: the sanitizer and its tests
    legitimately contain ``/home/<user>/`` strings as *data* to be redacted.
    Only strings handed to a path-consuming call are operational dependencies.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func) not in _FILESYSTEM_CALLS:
            continue
        for argument in node.args:
            if (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and FORBIDDEN_PATH_PATTERN.search(argument.value)
            ):
                offenders.append(argument.value[:160])
    return offenders


def test_no_operator_absolute_paths_used_as_filesystem_targets() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _scanned_files():
        if path.name == "test_packaging_and_paths.py":
            continue  # necessarily contains the pattern it forbids
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".py":
            hits = _filesystem_path_literals(text)
        else:
            hits = [
                line.strip()[:160]
                for line in text.splitlines()
                if not line.lstrip().startswith(("#", ";", "//"))
                and FORBIDDEN_PATH_PATTERN.search(line)
            ]
        if hits:
            offenders[str(path.relative_to(COMPONENT_ROOT))] = hits
    assert not offenders, f"operator-machine paths used as filesystem targets: {offenders}"


def test_pytest_ini_pythonpath_is_repository_relative() -> None:
    parser = configparser.ConfigParser()
    parser.read(COMPONENT_ROOT / "pytest.ini")
    entries = [line.strip() for line in parser["pytest"]["pythonpath"].splitlines() if line.strip()]
    assert entries, "pythonpath must not be empty"
    absolute = [entry for entry in entries if Path(entry).is_absolute()]
    assert not absolute, f"pytest.ini pythonpath must be relative; got absolute: {absolute}"
    for entry in entries:
        assert (COMPONENT_ROOT / entry).is_dir(), f"pythonpath entry does not exist: {entry}"


def _pyproject() -> dict:
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.loads((COMPONENT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytest.skip("tomllib requires Python 3.11+; packaging metadata asserted on newer interpreters")


def test_runtime_dependency_on_jsonschema_is_declared() -> None:
    """ingot_schema imports jsonschema on every cast, so it cannot be optional."""
    data = _pyproject()
    declared = " ".join(data["project"]["dependencies"]).lower()
    assert "jsonschema" in declared


def test_optional_parquet_and_zstd_extras_exist() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "pyarrow" in " ".join(extras["parquet"]).lower()
    assert "zstandard" in " ".join(extras["zstd"]).lower()
    assert "pytest" in " ".join(extras["dev"]).lower()


def test_heavy_optional_dependencies_stay_out_of_base_install() -> None:
    base = " ".join(_pyproject()["project"]["dependencies"]).lower()
    for heavy in ("pyarrow", "zstandard", "pandas", "numpy", "torch"):
        assert heavy not in base, f"{heavy} must not be a base dependency"


def test_cli_entry_point_is_declared_and_importable() -> None:
    scripts = _pyproject()["project"]["scripts"]
    assert scripts["goldtrace-refinery"] == "goldtrace_refinery.cli:main"
    from goldtrace_refinery.cli import main

    assert callable(main)


def test_packaged_contract_matches_canonical_contract() -> None:
    """The wheel-bundled schema must not drift from the repository contract."""
    for filename in ("ingot.v1.schema.json", "foundry-trust-policy.v1.schema.json"):
        canonical = (COMPONENT_ROOT / "contracts" / filename).read_bytes()
        packaged = (
            COMPONENT_ROOT / "src" / "goldtrace_refinery" / "contracts" / filename
        ).read_bytes()
        assert packaged == canonical, f"package-data {filename} has drifted from contracts/"
    # There is no longer a third copy under donor_components/. goldtrace_hallmark
    # and goldtrace_vault were promoted into src/, so the only two locations are
    # the canonical contracts/ and the package data that ships beside it. Full
    # coverage of both lives in
    # test_hallmark_contracts.py::test_shipped_contracts_have_not_drifted_from_the_repository.
    assert not (COMPONENT_ROOT / "donor_components").exists(), (
        "donor_components/ is back; a third contract copy will drift"
    )


def test_contract_lookup_resolves_without_environment_help(monkeypatch) -> None:
    from goldtrace_refinery import paths

    monkeypatch.delenv(paths.CONTRACTS_ENV_VAR, raising=False)
    assert paths.find_contract("ingot.v1.schema.json").is_file()


def test_interactive_contract_resolves_through_the_shared_lookup(monkeypatch) -> None:
    """Regression: _contract() once built a superproject-relative path.

    ``Path(__file__).resolve().parents[3] / "contracts"`` resolved to a directory
    that exists in no installed release and in no clone of this component alone,
    so casting a live_interactive bundle failed outside one operator's checkout.
    """
    from goldtrace_refinery import interactive, paths

    monkeypatch.delenv(paths.CONTRACTS_ENV_VAR, raising=False)
    assert interactive._contract()["$id"] == interactive.SCHEMA_ID

    source = (
        COMPONENT_ROOT / "src" / "goldtrace_refinery" / "interactive.py"
    ).read_text(encoding="utf-8")
    assert "parents[3]" not in source, (
        "interactive.py is reaching outside the component again; contracts must "
        "resolve through paths.find_contract()"
    )


def test_missing_contract_reports_every_location_tried() -> None:
    from goldtrace_refinery.paths import find_contract

    with pytest.raises(FileNotFoundError) as excinfo:
        find_contract("no-such-contract.v9.schema.json")
    assert "Tried:" in str(excinfo.value)


def test_optional_packager_backends_are_guarded_not_assumed() -> None:
    """packager must expose availability flags rather than importing eagerly."""
    source = (
        COMPONENT_ROOT / "src" / "goldtrace_mint" / "factory" / "packager.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    guarded = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"HAS_PYARROW", "HAS_ZSTD"}
    }
    assert guarded == {"HAS_PYARROW", "HAS_ZSTD"}
