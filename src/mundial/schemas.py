"""Contratos publicos y serializables del predictor y simulador."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MatchPrediction:
    team_a: str
    team_b: str
    prob_a: float
    prob_draw: float
    prob_b: float
    expected_goals_a: float
    expected_goals_b: float
    likely_score: tuple[int, int]

    def __post_init__(self) -> None:
        probabilities = (self.prob_a, self.prob_draw, self.prob_b)
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("Las probabilidades deben estar entre 0 y 1")
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("Las probabilidades deben sumar 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroupProjection:
    team: str
    expected_points: float
    expected_goals_for: float
    expected_goals_against: float
    qualification_probability: float


@dataclass(frozen=True)
class BracketMatch:
    match_id: str
    round_name: str
    team_a: str
    team_b: str
    winner: str
    probability_a: float
    probability_b: float
    forced: bool = False


@dataclass
class TournamentSimulation:
    group_tables: dict[str, list[GroupProjection]]
    bracket: list[BracketMatch]
    champion_probabilities: dict[str, float]
    runs: int
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

