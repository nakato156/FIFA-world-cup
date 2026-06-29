"""Carga del modelo seleccionado y adaptacion al contrato de prediccion."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np

from mundial.config import ARTIFACTS_DIR
from mundial.data import STATIC_FEATURES
from mundial.schemas import MatchPrediction


class KerasPredictor:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        import tensorflow as tf

        self.model = tf.keras.models.load_model(artifacts_dir / "selected_model.keras")
        bundle = joblib.load(artifacts_dir / "inference_bundle.joblib")
        self.imputer = bundle["imputer"]
        self.scaler = bundle["scaler"]
        self.team_states = bundle["team_states"]
        self.team_sequences = bundle["team_sequences"]
        self.model_type = bundle["model_type"]
        self.default_state = bundle["default_state"]

    def _state(self, team: str) -> dict[str, float]:
        return self.team_states.get(team, self.default_state)

    def _raw_features(self, team_a: str, team_b: str) -> list[float]:
        a, b = self._state(team_a), self._state(team_b)
        feature_map = {
            "rank_a": a["rank"], "rank_b": b["rank"], "rank_diff": b["rank"] - a["rank"],
            "goals_for_last10_a": a["goals_for_last10"], "goals_against_last10_a": a["goals_against_last10"],
            "goals_for_last10_b": b["goals_for_last10"], "goals_against_last10_b": b["goals_against_last10"],
            "h2h_wins_a": 0.0, "h2h_draws": 0.0, "h2h_wins_b": 0.0, "h2h_goal_difference": 0.0,
            "players_imputed_a": a["players_imputed"], "players_imputed_b": b["players_imputed"], "neutral": 1.0,
        }
        for attribute in ("overall", "pace", "shooting", "defending", "physical"):
            feature_map[f"{attribute}_a"] = a[attribute]
            feature_map[f"{attribute}_b"] = b[attribute]
        return [feature_map[name] for name in STATIC_FEATURES]

    def predict_matches(self, pairs: list[tuple[str, str]]) -> list[MatchPrediction]:
        """Predice muchos cruces en un unico batch para Monte Carlo."""
        if any(team_a == team_b for team_a, team_b in pairs):
            raise ValueError("Una seleccion no puede jugar contra si misma")
        raw = np.asarray([self._raw_features(team_a, team_b) for team_a, team_b in pairs], dtype=np.float32)
        static = self.scaler.transform(self.imputer.transform(raw)).astype(np.float32)
        if self.model_type.startswith("mlp"):
            inputs = static
        else:
            zero_sequence = np.zeros((10, 5), dtype=np.float32)
            inputs = {
                "static": static,
                "sequence_a": np.stack([self.team_sequences.get(team_a, zero_sequence) for team_a, _ in pairs]),
                "sequence_b": np.stack([self.team_sequences.get(team_b, zero_sequence) for _, team_b in pairs]),
            }
        result, goals_a, goals_b = self.model.predict(inputs, verbose=0)
        predictions: list[MatchPrediction] = []
        for index, (team_a, team_b) in enumerate(pairs):
            probabilities = np.asarray(result[index], dtype=float)
            probabilities /= probabilities.sum()
            expected_a = float(goals_a[index, 0])
            expected_b = float(goals_b[index, 0])
            predictions.append(MatchPrediction(
                team_a=team_a, team_b=team_b,
                prob_a=float(probabilities[0]), prob_draw=float(probabilities[1]), prob_b=float(probabilities[2]),
                expected_goals_a=expected_a, expected_goals_b=expected_b,
                likely_score=(max(0, math.floor(expected_a)), max(0, math.floor(expected_b))),
            ))
        return predictions

    def predict_match(self, team_a: str, team_b: str) -> MatchPrediction:
        return self.predict_matches([(team_a, team_b)])[0]


def load_predictor(artifacts_dir: Path = ARTIFACTS_DIR):
    """Usa el modelo real cuando existe y deja un fallback explicito para demo."""
    from mundial.prediction import DemoPredictor

    if (artifacts_dir / "selected_model.keras").exists() and (artifacts_dir / "inference_bundle.joblib").exists():
        return KerasPredictor(artifacts_dir), "Modelo neuronal entrenado"
    return DemoPredictor(), "Modo demostracion (entrene los modelos para usar IA real)"


def predict_match(team_a: str, team_b: str, artifacts_dir: Path = ARTIFACTS_DIR) -> MatchPrediction:
    """Interfaz funcional publica para predecir cualquier par de selecciones."""
    predictor, _ = load_predictor(artifacts_dir)
    return predictor.predict_match(team_a, team_b)
