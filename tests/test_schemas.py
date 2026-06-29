import pytest

from mundial.schemas import MatchPrediction
from mundial.statistical import align_score_matrix, dixon_coles_matrix, outcome_probabilities


def test_prediction_requires_probabilities_to_sum_one():
    with pytest.raises(ValueError, match="sumar 1"):
        MatchPrediction("A", "B", 0.5, 0.3, 0.3, 1.0, 1.0, (1, 1))


def test_prediction_accepts_valid_distribution():
    result = MatchPrediction("A", "B", 0.5, 0.25, 0.25, 1.5, 0.8, (1, 0))
    assert result.likely_score == (1, 0)


def test_score_matrix_is_single_source_of_probability_truth():
    matrix, retained = dixon_coles_matrix(1.6, 0.9, rho=-0.04)
    assert retained > 1.0 - 1e-6
    target = (0.55, 0.25, 0.20)
    aligned = align_score_matrix(matrix, target)
    prediction = MatchPrediction.from_score_matrix("A", "B", aligned)
    assert outcome_probabilities(aligned) == pytest.approx(target)
    assert prediction.prob_a == pytest.approx(target[0])
    assert prediction.score_probabilities is not None


def test_score_matrix_rejects_incoherent_summary():
    matrix, _ = dixon_coles_matrix(1.2, 1.0)
    with pytest.raises(ValueError, match="no reproduce"):
        MatchPrediction("A", "B", 0.8, 0.1, 0.1, 1.2, 1.0, (1, 0), tuple(map(tuple, matrix)))
