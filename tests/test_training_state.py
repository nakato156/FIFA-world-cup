import numpy as np
import pandas as pd

from mundial.training import _team_inference_state


def test_inference_state_contains_last_completed_match_and_h2h():
    frame = pd.DataFrame([{
        "date": pd.Timestamp("2026-01-01"), "match_id": 0,
        "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 1,
        "neutral": True, "rank_a": 10, "rank_b": 20,
        **{f"{attribute}_a": 70 for attribute in ("overall", "pace", "shooting", "defending", "physical")},
        **{f"{attribute}_b": 65 for attribute in ("overall", "pace", "shooting", "defending", "physical")},
        "players_imputed_a": 0, "players_imputed_b": 0,
    }])
    sequences = np.zeros((1, 10, 5), dtype=np.float32)
    states, latest, _, h2h = _team_inference_state(frame, sequences, sequences)
    assert states["A"]["goals_for_last10"] == 2
    assert latest["A"][-1].tolist() == [2.0, 1.0, 1.0, 0.0, 1.0]
    assert h2h[("A", "B")]["wins_first"] == 1
