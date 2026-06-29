import numpy as np
import pandas as pd

from mundial.data import SEQUENCE_FEATURES, _rolling_team_history, build_sequences, load_results


def test_temporal_window_excludes_current_match():
    matches = pd.DataFrame(
        {
            "match_id": [0, 1, 2],
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "C", "D"],
            "home_score": [1, 4, 2],
            "away_score": [0, 1, 2],
            "neutral": [True, True, True],
        }
    )
    history = _rolling_team_history(matches)
    seq_a, _ = build_sequences(matches, history, lookback=10)
    assert seq_a.shape == (3, 10, len(SEQUENCE_FEATURES))
    assert np.all(seq_a[0] == 0)
    assert seq_a[1, -1, 0] == 1
    assert seq_a[2, -1, 0] == 4
    first_history = history[(history["match_id"] == 0) & (history["team"] == "A")].iloc[0]
    second_history = history[(history["match_id"] == 1) & (history["team"] == "A")].iloc[0]
    assert np.isnan(first_history["goals_for_last10"])
    assert second_history["goals_for_last10"] == 1


def test_results_respect_as_of_date(tmp_path):
    pd.DataFrame({
        "date": ["2025-01-01", "2026-01-01"],
        "home_team": ["A", "A"], "away_team": ["B", "C"],
        "home_score": [1, 2], "away_score": [0, 0],
    }).to_csv(tmp_path / "results.csv", index=False)
    result = load_results(tmp_path, as_of_date="2025-06-01")
    assert result["date"].tolist() == [pd.Timestamp("2025-01-01")]
