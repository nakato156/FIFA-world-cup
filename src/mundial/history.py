"""Contexto historico de campeones con respaldo para modo demo."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from mundial.config import RAW_DIR
from mundial.data import normalize_team

FALLBACK_TITLES = {
    "Brazil": 5, "Germany": 4, "Italy": 4, "Argentina": 3,
    "France": 2, "Uruguay": 2, "England": 1, "Spain": 1,
}


def load_world_cup_titles() -> tuple[dict[str, int], str]:
    candidates = sorted(RAW_DIR.rglob("*.csv"))
    for path in candidates:
        try:
            frame = pd.read_csv(path, nrows=100)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        winner_column = next((column for column in frame if column.lower() in {"winner", "champion"}), None)
        if winner_column is None:
            continue
        winners = [normalize_team(value) for value in frame[winner_column].dropna()]
        if 5 <= len(winners) <= 30:
            return dict(Counter(winners)), f"Dataset: {path.name}"
    return dict(FALLBACK_TITLES), "Resumen historico incorporado; descargue Kaggle para usar la fuente obligatoria"

