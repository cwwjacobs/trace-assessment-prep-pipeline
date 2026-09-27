from __future__ import annotations

import fcntl
import hashlib
import json
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


HALLMARK_REPORT_NAME = "HALLMARK_REPORT.json"
PRODUCT_MANIFEST_NAME = "product-manifest.json"
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_REQUIRED_HALLMARK_PASS_CHECKS = frozenset(
    {
        "exists:dataset/train.jsonl",
        "exists:dataset/manifest.json",
        "exists:eval/pack/cases.jsonl",
        "exists:eval/pack/answers.jsonl",
        "exists:eval/graders/grader.json",
        "exists:eval/manifest.json",
        "exists:curation/decisions.jsonl",
        "exists:curation/rubric.json",
        "exists:provenance/source-ingots.jsonl",
        "exists:provenance/lineage.jsonl",
        "exists:provenance/rights.json",
        "exists:product-spec.json",
        "exists:product-manifest.json",
        "exists:SHA256SUMS",
        "product_manifest_schema",
        "product_lot_stability",
        "train_jsonl_parse",
        "decisions_jsonl_parse",
        "source_ingots_parse",
        "train_count_match",
        "included_have_decisions",
        "excluded_not_in_train",
        "source_ingot_resolvable",
        "eval_cases_loadable",
        "eval_answers_loadable",
        "no_split_leakage",
        "eval_case_answer_integrity",
        "eval_manifest_case_count",
        "product_hash_binding",
        "checksums",
        "dataset_manifest_binding",
        "eval_manifest_binding",
        "product_spec_hash_binding",
        "curation_ledger_hash_binding",
        "source_ingot_ledger_hash_binding",
        "lineage_manifest_hash_binding",
        "rights_manifest_hash_binding",
        "train_payload_hash_binding",
        "validation_payload_hash_binding",
        "curation_decision_integrity",
        "dataset_eval_version_correspondence",
        "rights_metadata_present",
        "privacy_requirements_declared",
        "eval_grader_contract",
        "semantic_rereview",
        "semantic_review_attribution",
        "semantic_reviewer_scope",
        "production_mode_policy",
        "eval_pack_present",
        "curation_decisions_present",
        # Substance. Every other name in this set is satisfied by a product
        # containing nothing: empty payloads match the hash of empty payloads,
        # parse without a bad line, and declare a count of zero their manifest
        # agrees with. Only these two require that anything is actually there.
        #
        # Just the unconditional pair. Hallmark emits substance.min_turns_per_record
        # and substance.required_roles_present only when a rubric asks for them,
        # so requiring those here would reject every lot whose rubric stayed
        # silent — a check that did not run must not be demanded.
        "substance.train_not_empty",
        "substance.train_min_records",
    }
)


def _schema_path(filename: str) -> Path:
    """Resolve a contract from package data when installed, or the checkout in dev.

    A fixed number of parent directories only describes one source layout. The
    Vault decides what may be sold, so it has to resolve its contracts from an
    installed release too — and an unimportable admission gate is how entries
    came to be written into the vault by a script instead.
    """
    from goldtrace_refinery.paths import find_contract

    return find_contract(filename)


class VaultError(Exception):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(obj: Any) -> bytes:
    try:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VaultError(f"value is not canonical JSON: {exc}") from exc


def _sha256_json(obj: Any) -> str:
    return _sha256_bytes(_canonical_json(obj))


def _object_without_key(obj: dict[str, Any], key: str) -> dict[str, Any]:
    normalized = deepcopy(obj)
    normalized.pop(key, None)
    return normalized


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise VaultError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise VaultError(f"invalid {label}: expected a JSON object")
    return value


@lru_cache(maxsize=1)
def _hallmark_report_validator() -> Draft202012Validator:
    schema = _load_json_object(_schema_path("hallmark-report.v1.schema.json"), label="Hallmark report schema")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise VaultError(f"invalid Hallmark report schema: {exc.message}") from exc
    return Draft202012Validator(schema)


def _validate_hallmark_report_schema(report: dict[str, Any]) -> None:
    errors = sorted(
        _hallmark_report_validator().iter_errors(report),
        key=lambda error: error.json_path,
    )
    if errors:
        error = errors[0]
        raise VaultError(
            f"Hallmark report schema validation failed at {error.json_path}: {error.message}"
        )


@lru_cache(maxsize=1)
def _product_manifest_validator() -> Draft202012Validator:
    schema = _load_json_object(_schema_path("product-lot.v1.schema.json"), label="product lot schema")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise VaultError(f"invalid product lot schema: {exc.message}") from exc
    return Draft202012Validator(schema)


def _validate_product_manifest_schema(manifest: dict[str, Any]) -> None:
    errors = sorted(
        _product_manifest_validator().iter_errors(manifest),
        key=lambda error: error.json_path,
    )
    if errors:
        error = errors[0]
        raise VaultError(
            f"product manifest schema validation failed at {error.json_path}: {error.message}"
        )


@lru_cache(maxsize=1)
def _vault_entry_validator() -> Draft202012Validator:
    schema = _load_json_object(_schema_path("vault-entry.v1.schema.json"), label="Vault entry schema")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise VaultError(f"invalid Vault entry schema: {exc.message}") from exc
    return Draft202012Validator(schema)


def _validate_vault_entry_schema(entry: dict[str, Any]) -> None:
    errors = sorted(
        _vault_entry_validator().iter_errors(entry),
        key=lambda error: error.json_path,
    )
    if errors:
        error = errors[0]
        raise VaultError(
            f"Vault entry schema validation failed at {error.json_path}: {error.message}"
        )


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _resolve_directory(path: Path, *, label: str) -> Path:
    original = Path(path)
    if original.is_symlink():
        raise VaultError(f"{label} must not be a symlink")
    try:
        resolved = original.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise VaultError(f"{label} does not resolve safely: {exc}") from exc
    if not resolved.is_dir():
        raise VaultError(f"{label} is not a directory: {resolved}")
    return resolved


def _resolve_report_file(path: Path) -> Path:
    original = Path(path)
    if original.is_symlink():
        raise VaultError("Hallmark report path must not be a symlink")
    try:
        resolved = original.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise VaultError(f"Hallmark report path does not resolve safely: {exc}") from exc
    try:
        mode = resolved.lstat().st_mode
    except OSError as exc:
        raise VaultError(f"cannot inspect Hallmark report path: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise VaultError("Hallmark report path must be a regular file")
    return resolved


def _lot_files(root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    try:
        paths = list(root.rglob("*"))
    except OSError as exc:
        raise VaultError(f"cannot traverse product lot: {exc}") from exc

    for path in paths:
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise VaultError(f"cannot inspect product lot path {path}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise VaultError(f"product lot contains a symlink: {path.relative_to(root)}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise VaultError(f"product lot contains a non-regular file: {path.relative_to(root)}")

        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise VaultError(f"product lot path does not resolve safely: {path}: {exc}") from exc
        if not _is_within(resolved, root):
            raise VaultError(f"product lot path escapes its root: {path}")

        relative = path.relative_to(root).as_posix()
        if "\n" in relative or "\r" in relative:
            raise VaultError("product lot paths must not contain line breaks")
        files.append((relative, path))

    return sorted(files, key=lambda item: item[0])


def _product_lot_hash(
    root: Path,
    *,
    allow_vault_report: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Hash an exact lot without making product_hash self-referential.

    Every regular product-lot file contributes a sorted ledger line. The
    product manifest contributes canonical JSON with only product_hash
    omitted. A Vault-owned Hallmark report is not part of the source lot and
    is ignored only while revalidating an already admitted object.
    """

    manifest_path = root / PRODUCT_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise VaultError(f"missing or unsafe {PRODUCT_MANIFEST_NAME}")
    manifest = _load_json_object(manifest_path, label=PRODUCT_MANIFEST_NAME)

    lines: list[str] = []
    saw_manifest = False
    for relative, path in _lot_files(root):
        if relative == HALLMARK_REPORT_NAME:
            if allow_vault_report:
                continue
            raise VaultError(
                f"product lot contains reserved Vault attachment {HALLMARK_REPORT_NAME}"
            )
        if relative == PRODUCT_MANIFEST_NAME:
            saw_manifest = True
            content_hash = _sha256_bytes(
                _canonical_json(_object_without_key(manifest, "product_hash"))
            )
        else:
            content_hash = _sha256_file(path)
        lines.append(f"{content_hash}  {relative}")

    if not saw_manifest:
        raise VaultError(f"missing {PRODUCT_MANIFEST_NAME}")
    ledger = ("\n".join(lines) + "\n").encode("utf-8")
    return _sha256_bytes(ledger), manifest


def _required_nonempty_string(obj: dict[str, Any], field: str, *, label: str) -> str:
    value = obj.get(field)
    if not isinstance(value, str) or not value:
        raise VaultError(f"{label} must contain a nonempty {field}")
    return value


def _required_sha256(obj: dict[str, Any], field: str, *, label: str) -> str:
    value = _required_nonempty_string(obj, field, label=label)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise VaultError(f"{label} {field} must be a lowercase SHA-256 digest")
    return value


def _validated_lot(root: Path) -> tuple[dict[str, Any], str]:
    actual_hash, manifest = _product_lot_hash(root)
    _validate_product_manifest_schema(manifest)
    _required_nonempty_string(manifest, "product_id", label="product manifest")
    _required_nonempty_string(manifest, "product_version", label="product manifest")
    declared_hash = _required_sha256(manifest, "product_hash", label="product manifest")
    if declared_hash != actual_hash:
        raise VaultError(
            "product manifest product_hash does not match the exact product lot"
        )
    return manifest, actual_hash


def _validated_rights_status(root: Path, manifest: dict[str, Any]) -> str:
    rights_path = root / "provenance" / "rights.json"
    declared_hash = manifest.get("rights_manifest_hash")
    if not rights_path.exists():
        if declared_hash:
            raise VaultError("product manifest binds a missing provenance/rights.json")
        return "unknown"
    if rights_path.is_symlink() or not rights_path.is_file():
        raise VaultError("provenance/rights.json must be a regular file")
    expected_hash = _required_sha256(
        manifest,
        "rights_manifest_hash",
        label="product manifest",
    )
    if _sha256_file(rights_path) != expected_hash:
        raise VaultError("rights_manifest_hash does not match provenance/rights.json")
    rights = _load_json_object(rights_path, label="provenance/rights.json")
    status_value = rights.get("rights_status")
    if status_value == "fixture":
        return "fixture"
    if status_value == "declared":
        return "declared"
    return "unknown"


def _validated_report_hash(report: dict[str, Any]) -> str:
    declared = _required_sha256(report, "report_hash", label="Hallmark report")
    recomputed = _sha256_json(_object_without_key(report, "report_hash"))
    if declared != recomputed:
        raise VaultError("Hallmark report_hash integrity verification failed")
    return declared


def _validate_pass_report_claims(report: dict[str, Any]) -> None:
    """Reject a contradictory or incomplete self-declared Hallmark PASS.

    This is a structural trust floor, not reviewer authentication. A future
    operator-owned signature is still required to authenticate who issued the
    report.
    """

    checks = report.get("checks")
    if not isinstance(checks, list):
        raise VaultError("Hallmark PASS checks must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("name"), str):
            raise VaultError("Hallmark PASS contains a malformed check")
        name = check["name"]
        if name in by_name:
            raise VaultError(f"Hallmark PASS contains duplicate check {name!r}")
        by_name[name] = check
        if check.get("ok") is not True:
            raise VaultError(f"Hallmark PASS contradicts failed check {name!r}")
    missing = sorted(_REQUIRED_HALLMARK_PASS_CHECKS - set(by_name))
    if missing:
        raise VaultError(f"Hallmark PASS lacks required trust-floor checks: {missing}")
    if report.get("discrepancies") != []:
        raise VaultError("Hallmark PASS must not retain unresolved discrepancies")
    semantic_reviews = report.get("semantic_reviews")
    if not isinstance(semantic_reviews, list) or not semantic_reviews:
        raise VaultError("Hallmark PASS lacks retained semantic reviewer dispositions")
    if any(
        not isinstance(review, dict) or review.get("disposition") != "ACCEPT"
        for review in semantic_reviews
    ):
        raise VaultError("Hallmark PASS contradicts a non-ACCEPT semantic disposition")


class LocalVault:
    def __init__(self, root: Path, *, production_mode: bool = False):
        original_root = Path(root)
        if original_root.is_symlink():
            raise VaultError("Vault root must not be a symlink")
        self.root = original_root.resolve()
        self.production_mode = production_mode
        self.objects = self.root / "objects"
        self.ledger = self.root / "admission.jsonl"
        self.lock_file = self.root / ".admission.lock"

        self.root.mkdir(parents=True, exist_ok=True)
        if self.objects.is_symlink():
            raise VaultError("Vault objects directory must not be a symlink")
        self.objects.mkdir(parents=True, exist_ok=True)
        if self.objects.resolve() != self.root / "objects":
            raise VaultError("Vault objects directory escapes the Vault root")

        for path, label in (
            (self.ledger, "Vault ledger"),
            (self.lock_file, "Vault admission lock"),
        ):
            if path.is_symlink():
                raise VaultError(f"{label} must not be a symlink")
            if path.exists() and not path.is_file():
                raise VaultError(f"{label} must be a regular file")
            path.touch(exist_ok=True)

    def _assert_internal_paths(self) -> None:
        if self.objects.is_symlink() or self.objects.resolve() != self.root / "objects":
            raise VaultError("Vault objects directory escapes the Vault root")
        for path, label in (
            (self.ledger, "Vault ledger"),
            (self.lock_file, "Vault admission lock"),
        ):
            if path.is_symlink() or not path.is_file():
                raise VaultError(f"{label} is not a safe regular file")

    @contextmanager
    def _admission_lock(self) -> Iterator[None]:
        self._assert_internal_paths()
        with self.lock_file.open("r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _ledger_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            lines = self.ledger.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise VaultError(f"cannot read Vault ledger: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except (json.JSONDecodeError, ValueError) as exc:
                raise VaultError(f"invalid Vault ledger row {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise VaultError(f"invalid Vault ledger row {line_number}: expected object")
            rows.append(row)
        return rows

    def _load_report(
        self,
        hallmark_report: dict[str, Any] | Path,
        *,
        product_lot: Path,
    ) -> dict[str, Any]:
        if isinstance(hallmark_report, Path):
            report_path = _resolve_report_file(hallmark_report)
            if _is_within(report_path, product_lot):
                raise VaultError("Hallmark report path must be outside the product lot")
            if _is_within(report_path, self.root):
                raise VaultError("Hallmark report path must be outside the Vault root")
            report = _load_json_object(report_path, label="Hallmark report")
        elif isinstance(hallmark_report, dict):
            report = deepcopy(hallmark_report)
        else:
            raise VaultError("Hallmark report must be a JSON object or a Path")
        _validate_hallmark_report_schema(report)
        return report

    def _validate_existing_object(
        self,
        *,
        destination: Path,
        manifest: dict[str, Any],
        product_hash: str,
        report: dict[str, Any],
        report_hash: str,
        ledger_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if destination.is_symlink() or not destination.is_dir():
            raise VaultError("overwrite refused: existing Vault object is unsafe")
        try:
            existing_hash, existing_manifest = _product_lot_hash(
                destination,
                allow_vault_report=True,
            )
        except VaultError as exc:
            raise VaultError(
                f"overwrite refused: existing Vault object failed integrity verification: {exc}"
            ) from exc
        if existing_hash != product_hash:
            raise VaultError("overwrite refused: existing Vault object content changed")
        if (
            existing_manifest.get("product_hash") != product_hash
            or existing_manifest.get("product_id") != manifest.get("product_id")
            or existing_manifest.get("product_version") != manifest.get("product_version")
        ):
            raise VaultError("overwrite refused: existing Vault object identity changed")

        stored_report_path = destination / HALLMARK_REPORT_NAME
        if stored_report_path.is_symlink() or not stored_report_path.is_file():
            raise VaultError("overwrite refused: stored Hallmark report is missing or unsafe")
        stored_report = _load_json_object(stored_report_path, label="stored Hallmark report")
        _validate_hallmark_report_schema(stored_report)
        stored_report_hash = _validated_report_hash(stored_report)
        if stored_report_hash != report_hash or stored_report != report:
            raise VaultError("overwrite refused: a different Hallmark report is already stored")

        matching_rows = [
            row
            for row in ledger_rows
            if row.get("action") == "admit"
            and row.get("product_lot_hash") == product_hash
        ]
        if len(matching_rows) != 1:
            raise VaultError("overwrite refused: Vault object/ledger state is inconsistent")
        row = matching_rows[0]
        if (
            row.get("product_id") != manifest.get("product_id")
            or row.get("product_version") != manifest.get("product_version")
            or row.get("hallmark_report_hash") != report_hash
            or Path(str(row.get("storage_location"))).resolve() != destination
        ):
            raise VaultError("overwrite refused: Vault admission receipt is inconsistent")
        return {
            "status": "already_admitted",
            "product_id": manifest["product_id"],
            "product_version": manifest["product_version"],
            "storage_location": str(destination),
            "product_lot_hash": product_hash,
            "hallmark_report_hash": report_hash,
        }

    def admit(
        self,
        product_lot: Path,
        hallmark_report: dict[str, Any] | Path,
    ) -> dict[str, Any]:
        product_lot = _resolve_directory(Path(product_lot), label="product lot")
        if _paths_overlap(product_lot, self.root):
            raise VaultError("product lot and Vault root must not overlap")

        report = self._load_report(hallmark_report, product_lot=product_lot)
        manifest, product_hash = _validated_lot(product_lot)
        rights_status = _validated_rights_status(product_lot, manifest)

        if report.get("status") != "PASS":
            raise VaultError(f"Vault refuses non-PASS Hallmark status: {report.get('status')}")
        _validate_pass_report_claims(report)

        manifest_id = manifest["product_id"]
        manifest_version = manifest["product_version"]
        if report.get("product_id") != manifest_id:
            raise VaultError("Hallmark report refers to a different product_id")
        if report.get("product_version") != manifest_version:
            raise VaultError("Hallmark report refers to a different product_version")
        if report.get("product_hash") != product_hash:
            raise VaultError("Hallmark report refers to a different product_hash")
        if (
            "fixture_only" in report
            and report["fixture_only"] != bool(manifest.get("fixture_only"))
        ):
            raise VaultError("Hallmark report fixture_only does not match the product lot")

        # The contract's self-hash proves integrity and exact binding. It does
        # not authenticate reviewer identity; no signature contract exists.
        report_hash = _validated_report_hash(report)

        if self.production_mode:
            reviewer = report.get("reviewer") or {}
            if reviewer.get("mode") == "fixture" or reviewer.get("type") == "fixture":
                raise VaultError(
                    "Vault admission using fixture/mock Hallmark reviewer in production mode"
                )
            if report.get("fixture_only") or manifest.get("fixture_only"):
                raise VaultError("fixture_only product cannot be admitted in production mode")

        destination = self.objects / product_hash
        if destination.parent.resolve() != self.objects.resolve():
            raise VaultError("Vault object destination escapes the objects directory")

        with self._admission_lock():
            ledger_rows = self._ledger_rows()
            if destination.exists() or destination.is_symlink():
                return self._validate_existing_object(
                    destination=destination,
                    manifest=manifest,
                    product_hash=product_hash,
                    report=report,
                    report_hash=report_hash,
                    ledger_rows=ledger_rows,
                )

            for row in ledger_rows:
                if row.get("action") != "admit":
                    continue
                if (
                    row.get("product_id") == manifest_id
                    and row.get("product_version") == manifest_version
                ):
                    if row.get("product_lot_hash") == product_hash:
                        raise VaultError(
                            "Vault ledger refers to a missing content-addressed object"
                        )
                    raise VaultError("conflicting product identity already admitted")

            staging_parent = Path(
                tempfile.mkdtemp(prefix=".admit-", dir=str(self.objects))
            )
            staged_object = staging_parent / "object"
            destination_created = False
            admission_complete = False
            try:
                # Preserve any source symlink encountered during a race so the
                # staged validation rejects it instead of following it.
                shutil.copytree(product_lot, staged_object, symlinks=True)
                staged_manifest, staged_hash = _validated_lot(staged_object)
                if (
                    staged_hash != product_hash
                    or staged_manifest.get("product_id") != manifest_id
                    or staged_manifest.get("product_version") != manifest_version
                ):
                    raise VaultError("product lot changed while admission was in progress")
                _validated_rights_status(staged_object, staged_manifest)
                (staged_object / HALLMARK_REPORT_NAME).write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                staged_object.rename(destination)
                destination_created = True

                fixture_only = bool(manifest.get("fixture_only"))
                entry = {
                    "schema_id": "goldtrace.vault.entry.v1",
                    "product_id": manifest_id,
                    "product_version": manifest_version,
                    "product_lot_hash": product_hash,
                    "hallmark_report_hash": report_hash,
                    "hallmark_status": "PASS",
                    "admitted_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "storage_location": str(destination),
                    "previous_version": None,
                    "supersession": None,
                    "rights_status": rights_status,
                    "sale_eligibility": "fixture_only" if fixture_only else "held",
                    "fixture_only": fixture_only,
                    "action": "admit",
                }
                entry["admission_receipt_hash"] = _sha256_json(entry)
                _validate_vault_entry_schema(entry)
                try:
                    with self.ledger.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(entry, sort_keys=True) + "\n")
                        handle.flush()
                except OSError:
                    shutil.rmtree(destination)
                    destination_created = False
                    raise
                admission_complete = True
                return entry
            except OSError as exc:
                raise VaultError(f"Vault admission failed: {exc}") from exc
            finally:
                if destination_created and not admission_complete and destination.exists():
                    # Only a destination created by this locked transaction can
                    # reach this branch.
                    shutil.rmtree(destination)
                shutil.rmtree(staging_parent, ignore_errors=True)

    def withdraw(
        self,
        product_id: str,
        product_version: str,
        reason: str,
        *,
        retracted_admission_receipt_hash: str | None = None,
    ) -> dict[str, Any]:
        """Append a withdrawal to the ledger. The admission row is never removed.

        A withdrawal that deleted the admission would erase the claim it is
        correcting. Recording both leaves the ledger append-only, so the reader
        can see what was asserted, that it was retracted, and why.

        The written row is validated like an admission is. This previously was
        not: ``reason`` is not a property the entry contract defines, three
        fields were null where it required strings, and nothing checked, so
        every withdrawal produced a row that violated its own contract.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise VaultError("a withdrawal must state a reason")
        if retracted_admission_receipt_hash is not None and not _SHA256_PATTERN.match(
            retracted_admission_receipt_hash
        ):
            raise VaultError(
                "retracted_admission_receipt_hash must be a sha256 digest or null"
            )
        entry = {
            "schema_id": "goldtrace.vault.entry.v1",
            "product_id": product_id,
            "product_version": product_version,
            "product_lot_hash": None,
            "hallmark_report_hash": None,
            # A withdrawn row asserting PASS would leave a false claim standing
            # in the ledger.
            "hallmark_status": "WITHDRAWN",
            "admitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "storage_location": None,
            "previous_version": product_version,
            "supersession": None,
            "rights_status": "withdrawn",
            "sale_eligibility": "withdrawn",
            "action": "withdraw",
            "reason": reason,
            "retracted_admission_receipt_hash": retracted_admission_receipt_hash,
        }
        entry["admission_receipt_hash"] = _sha256_json(entry)
        _validate_vault_entry_schema(entry)
        with self._admission_lock():
            with self.ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def publish(self, *args: Any, **kwargs: Any) -> None:
        raise VaultError("any automatic publication attempt is refused")

    def inspect(self, product_id: str) -> list[dict[str, Any]]:
        return [row for row in self._ledger_rows() if row.get("product_id") == product_id]
