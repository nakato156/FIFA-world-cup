"""Predictores para artefactos Keras y para demostracion sin datos."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Protocol

from mundial.schemas import MatchPrediction
from mundial.statistical import align_score_matrix, dixon_coles_matrix


class Predictor(Protocol):
    def predict_match(self, team_a: str, team_b: str, posterior_draw: int | None = None) -> MatchPrediction:
        """Predice un partido en cancha neutral."""


class DemoPredictor:
    """Baseline Elo determinista para probar el producto antes de entrenar."""

    def __init__(self, ratings_path: Path | Mapping[str, float] | None = None) -> None:
        self.ratings: dict[str, float] = {}
        if isinstance(ratings_path, Mapping):
            self.ratings = {team: float(value) for team, value in ratings_path.items()}
        elif ratings_path and ratings_path.exists():
            self.ratings = json.loads(ratings_path.read_text(encoding="utf-8"))

    @staticmethod
    def _stable_rating(team: str) -> float:
        digest = hashlib.sha256(team.encode("utf-8")).hexdigest()
        return 1450.0 + (int(digest[:8], 16) % 350)

    def predict_match(self, team_a: str, team_b: str, posterior_draw: int | None = None) -> MatchPrediction:
        if team_a == team_b:
            raise ValueError("Una seleccion no puede jugar contra si misma")
        rating_a = float(self.ratings.get(team_a, self._stable_rating(team_a)))
        rating_b = float(self.ratings.get(team_b, self._stable_rating(team_b)))
        strength_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
        draw = 0.22 + 0.08 * math.exp(-abs(rating_a - rating_b) / 180.0)
        prob_a = (1.0 - draw) * strength_a
        prob_b = (1.0 - draw) * (1.0 - strength_a)
        goals_a = max(0.2, 0.55 + 1.75 * strength_a)
        goals_b = max(0.2, 0.55 + 1.75 * (1.0 - strength_a))
        matrix, _ = dixon_coles_matrix(goals_a, goals_b)
        matrix = align_score_matrix(matrix, (prob_a, draw, prob_b))
        return MatchPrediction.from_score_matrix(team_a, team_b, matrix)
