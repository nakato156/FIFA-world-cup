import pytest

from mundial.config import load_groups
from mundial.prediction import DemoPredictor
from mundial.simulation import TournamentSimulator, validate_groups


def test_official_groups_are_valid():
    validate_groups(load_groups())


def test_duplicate_team_is_rejected():
    groups = load_groups()
    groups["B"][0] = groups["A"][0]
    with pytest.raises(ValueError, match="unicas"):
        validate_groups(groups)


def test_complete_tournament_has_expected_structure():
    result = TournamentSimulator(DemoPredictor()).simulate(load_groups(), runs=8, seed=7)
    assert len(result.bracket) == 32
    assert result.bracket[-1].match_id == "M104"
    assert result.metadata["group_matches_per_run"] == 72
    assert result.metadata["knockout_matches_per_run"] == 32
    assert sum(result.champion_probabilities.values()) == pytest.approx(1.0)
    qualification_sum = sum(
        row.qualification_probability
        for table in result.group_tables.values()
        for row in table
    )
    assert qualification_sum == pytest.approx(32.0)


def test_final_override_is_respected_with_same_seed():
    simulator = TournamentSimulator(DemoPredictor())
    baseline = simulator.simulate(load_groups(), runs=1, seed=11)
    final = baseline.bracket[-1]
    forced = final.team_b if final.winner == final.team_a else final.team_a
    result = simulator.simulate(load_groups(), overrides={"M104": forced}, runs=1, seed=11)
    assert result.bracket[-1].winner == forced
    assert result.bracket[-1].forced

