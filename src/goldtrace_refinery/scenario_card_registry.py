"""Scenario card pre-registration across Labyrinth, Foundry and Vault.

Reject-heavy by design: digs may vary (ore hashes change), but a product pack
id is only issued for a scenario card hash that the operator registered with
all three tiers before the dig. Unknown or partial registration fails closed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_ID = "gtdataworks.scenario_card_registry.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_TIERS = ("labyrinth", "foundry", "vault")


class ScenarioCardRegistryError(ValueError):
    """Card registry load or admission failure (always fail-closed)."""


@dataclass(frozen=True)
class RegisteredScenarioCard:
    scenario_id: str
    scenario_version: str
    card_hash: str
    tree_sha256: str | None
    product_pack_id: str
    admission_pack: str | None
    tiers: Mapping[str, bool]
    registered_at: str | None
    notes: str | None

    @property
    def card_id(self) -> str:
        return f"{self.scenario_id}@{self.scenario_version}"

    def fully_registered(self) -> bool:
        return all(bool(self.tiers.get(name)) for name in REQUIRED_TIERS)


@dataclass(frozen=True)
class ScenarioCardRegistry:
    cards: tuple[RegisteredScenarioCard, ...]
    registry_id: str | None = None
    updated_at: str | None = None
    path: Path | None = None

    def lookup(
        self,
        *,
        scenario_id: str,
        scenario_version: str | None = None,
        card_hash: str | None = None,
    ) -> RegisteredScenarioCard | None:
        matches = [
            card
            for card in self.cards
            if card.scenario_id == scenario_id
            and (scenario_version is None or card.scenario_version == scenario_version)
            and (card_hash is None or card.card_hash == card_hash)
        ]
        if len(matches) == 1:
            return matches[0]
        return None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScenarioCardRegistryError(message)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def load_scenario_card_registry(path: str | Path) -> ScenarioCardRegistry:
    """Load the operator scenario-card registry. Fail closed on any defect."""

    registry_path = Path(path).expanduser()
    if not registry_path.is_absolute():
        registry_path = Path.cwd() / registry_path
    if not registry_path.is_file() or registry_path.is_symlink():
        raise ScenarioCardRegistryError(
            f"scenario card registry missing or unsafe: {registry_path}"
        )
    try:
        document = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScenarioCardRegistryError(
            f"scenario card registry is unreadable or not JSON: {registry_path}: {exc}"
        ) from exc
    _require(isinstance(document, dict), "scenario card registry must be a JSON object")
    _require(
        document.get("schema_id") == SCHEMA_ID,
        f"scenario card registry schema_id must be {SCHEMA_ID}",
    )
    raw_cards = document.get("cards")
    _require(isinstance(raw_cards, list), "scenario card registry cards must be a list")
    cards: list[RegisteredScenarioCard] = []
    seen_hashes: set[str] = set()
    seen_products: set[str] = set()
    for entry in raw_cards:
        _require(isinstance(entry, dict), "each registered card must be an object")
        scenario_id = entry.get("scenario_id")
        scenario_version = entry.get("scenario_version")
        card_hash = entry.get("card_hash")
        product_pack_id = entry.get("product_pack_id")
        tiers = entry.get("tiers")
        tree_sha256 = entry.get("tree_sha256")
        _require(
            isinstance(scenario_id, str) and bool(scenario_id.strip()),
            "registered card scenario_id is required",
        )
        _require(
            isinstance(scenario_version, str) and bool(scenario_version.strip()),
            "registered card scenario_version is required",
        )
        _require(_is_sha256(card_hash), "registered card_hash must be lowercase SHA-256")
        _require(
            isinstance(product_pack_id, str) and bool(product_pack_id.strip()),
            "registered product_pack_id is required",
        )
        _require(isinstance(tiers, dict), "registered card tiers must be an object")
        _require(
            set(tiers) == set(REQUIRED_TIERS),
            f"registered card tiers must be exactly {list(REQUIRED_TIERS)}",
        )
        _require(
            all(isinstance(tiers[name], bool) for name in REQUIRED_TIERS),
            "registered card tier flags must be booleans",
        )
        if tree_sha256 is not None:
            _require(_is_sha256(tree_sha256), "tree_sha256 must be lowercase SHA-256 when set")
        if card_hash in seen_hashes:
            raise ScenarioCardRegistryError(
                f"duplicate card_hash in registry: {card_hash}"
            )
        if product_pack_id in seen_products:
            raise ScenarioCardRegistryError(
                f"duplicate product_pack_id in registry: {product_pack_id}"
            )
        seen_hashes.add(card_hash)
        seen_products.add(product_pack_id)
        admission_pack = entry.get("admission_pack")
        if admission_pack is not None:
            _require(
                isinstance(admission_pack, str) and bool(admission_pack.strip()),
                "admission_pack must be a non-empty string when set",
            )
        cards.append(
            RegisteredScenarioCard(
                scenario_id=scenario_id.strip(),
                scenario_version=scenario_version.strip(),
                card_hash=card_hash,
                tree_sha256=tree_sha256,
                product_pack_id=product_pack_id.strip(),
                admission_pack=admission_pack.strip() if isinstance(admission_pack, str) else None,
                tiers={name: bool(tiers[name]) for name in REQUIRED_TIERS},
                registered_at=entry.get("registered_at")
                if isinstance(entry.get("registered_at"), str)
                else None,
                notes=entry.get("notes") if isinstance(entry.get("notes"), str) else None,
            )
        )
    return ScenarioCardRegistry(
        cards=tuple(cards),
        registry_id=document.get("registry_id")
        if isinstance(document.get("registry_id"), str)
        else None,
        updated_at=document.get("updated_at")
        if isinstance(document.get("updated_at"), str)
        else None,
        path=registry_path,
    )


def read_bundle_scenario_identity(bundle: Path) -> dict[str, str]:
    """Read sealed scenario identity from a Labyrinth run bundle."""

    run_path = Path(bundle) / "run.json"
    if not run_path.is_file():
        raise ScenarioCardRegistryError(f"bundle has no run.json: {bundle}")
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScenarioCardRegistryError(f"bundle run.json unreadable: {exc}") from exc
    scenario = run.get("scenario")
    _require(isinstance(scenario, dict), "bundle run.json scenario block missing")
    scenario_id = scenario.get("id")
    version = scenario.get("version")
    card_hash = scenario.get("pack_sha256")
    tree_sha256 = scenario.get("tree_sha256")
    _require(
        isinstance(scenario_id, str) and scenario_id.strip(),
        "bundle scenario.id missing",
    )
    _require(
        isinstance(version, str) and version.strip(),
        "bundle scenario.version missing",
    )
    _require(_is_sha256(card_hash), "bundle scenario.pack_sha256 missing or not SHA-256")
    identity = {
        "scenario_id": scenario_id.strip(),
        "scenario_version": version.strip(),
        "card_hash": card_hash,
    }
    if isinstance(tree_sha256, str) and _is_sha256(tree_sha256):
        identity["tree_sha256"] = tree_sha256
    return identity


def require_registered_card(
    registry: ScenarioCardRegistry,
    *,
    scenario_id: str,
    scenario_version: str,
    card_hash: str,
    tree_sha256: str | None = None,
) -> RegisteredScenarioCard:
    """Reject-heavy admission: unknown, mismatched, or partial registration fails."""

    _require(_is_sha256(card_hash), "card_hash must be lowercase SHA-256")
    card = registry.lookup(
        scenario_id=scenario_id,
        scenario_version=scenario_version,
        card_hash=card_hash,
    )
    if card is None:
        # Distinguish id match with wrong hash from total unknown.
        by_id = registry.lookup(
            scenario_id=scenario_id, scenario_version=scenario_version
        )
        if by_id is not None:
            raise ScenarioCardRegistryError(
                f"scenario card hash is not the registered hash for "
                f"{scenario_id}@{scenario_version}: "
                f"bundle={card_hash} registered={by_id.card_hash}"
            )
        raise ScenarioCardRegistryError(
            f"scenario card is not registered for Foundry/Vault smelt: "
            f"{scenario_id}@{scenario_version} card_hash={card_hash}. "
            f"Register pre-run with tools/register_scenario_card.py "
            f"(reject-heavy: unknown cards do not smelt)."
        )
    if not card.fully_registered():
        missing = [name for name in REQUIRED_TIERS if not card.tiers.get(name)]
        raise ScenarioCardRegistryError(
            f"scenario card {card.card_id} is only partially registered; "
            f"missing tiers: {missing}. Reject-heavy policy requires "
            f"labyrinth + foundry + vault."
        )
    if (
        tree_sha256
        and card.tree_sha256
        and tree_sha256 != card.tree_sha256
    ):
        raise ScenarioCardRegistryError(
            f"scenario tree_sha256 mismatch for {card.card_id}: "
            f"bundle={tree_sha256} registered={card.tree_sha256}"
        )
    return card


def require_bundle_card(
    bundle: Path,
    registry: ScenarioCardRegistry,
) -> RegisteredScenarioCard:
    """Load sealed identity from a bundle and require full card registration."""

    identity = read_bundle_scenario_identity(bundle)
    return require_registered_card(
        registry,
        scenario_id=identity["scenario_id"],
        scenario_version=identity["scenario_version"],
        card_hash=identity["card_hash"],
        tree_sha256=identity.get("tree_sha256"),
    )


def labyrinth_registry_entry(
    labyrinth_root: Path,
    *,
    scenario_id: str,
    scenario_version: str,
) -> dict[str, Any]:
    """Read one entry from Labyrinth's scenarios/registry.json (source of card hash)."""

    path = Path(labyrinth_root) / "scenarios" / "registry.json"
    if not path.is_file():
        raise ScenarioCardRegistryError(f"Labyrinth scenario registry missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScenarioCardRegistryError(
            f"Labyrinth scenario registry unreadable: {exc}"
        ) from exc
    entries = document.get("scenarios")
    _require(isinstance(entries, list), "Labyrinth scenarios list missing")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("scenario_id") == scenario_id
        and entry.get("scenario_version") == scenario_version
    ]
    if len(matches) != 1:
        raise ScenarioCardRegistryError(
            f"Labyrinth registry lookup not unique for "
            f"{scenario_id}@{scenario_version}: matches={len(matches)}"
        )
    entry = matches[0]
    card_hash = entry.get("source_archive_sha256")
    _require(
        _is_sha256(card_hash),
        f"Labyrinth registry entry lacks source_archive_sha256 for "
        f"{scenario_id}@{scenario_version}",
    )
    return entry


def dump_registry(registry: ScenarioCardRegistry) -> dict[str, Any]:
    """Serialize a registry document for disk."""

    return {
        "schema_id": SCHEMA_ID,
        "registry_id": registry.registry_id,
        "updated_at": registry.updated_at,
        "cards": [
            {
                "scenario_id": card.scenario_id,
                "scenario_version": card.scenario_version,
                "card_hash": card.card_hash,
                "tree_sha256": card.tree_sha256,
                "product_pack_id": card.product_pack_id,
                "admission_pack": card.admission_pack,
                "tiers": {
                    "labyrinth": bool(card.tiers.get("labyrinth")),
                    "foundry": bool(card.tiers.get("foundry")),
                    "vault": bool(card.tiers.get("vault")),
                },
                "registered_at": card.registered_at,
                "notes": card.notes,
            }
            for card in registry.cards
        ],
    }
