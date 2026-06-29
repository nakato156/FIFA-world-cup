"""Carga de configuracion compartida por entrenamiento y dashboard."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"


def load_groups() -> dict[str, list[str]]:
    """Devuelve una copia de los 12 grupos oficiales configurados."""
    payload = json.loads((CONFIG_DIR / "groups_2026.json").read_text(encoding="utf-8"))
    return {group: list(teams) for group, teams in payload["groups"].items()}


def load_aliases() -> dict[str, str]:
    """Devuelve equivalencias de nombres usadas entre fuentes."""
    return json.loads((CONFIG_DIR / "team_aliases.json").read_text(encoding="utf-8"))


def ensure_runtime_dirs() -> None:
    """Crea solo directorios de datos generados y artefactos."""
    for directory in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

