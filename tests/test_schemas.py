import pytest

from mundial.schemas import MatchPrediction


def test_prediction_requires_probabilities_to_sum_one():
    with pytest.raises(ValueError, match="sumar 1"):
        MatchPrediction("A", "B", 0.5, 0.3, 0.3, 1.0, 1.0, (1, 1))


def test_prediction_accepts_valid_distribution():
    result = MatchPrediction("A", "B", 0.5, 0.25, 0.25, 1.5, 0.8, (1, 0))
    assert result.likely_score == (1, 0)

