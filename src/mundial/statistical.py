"""Calibracion bayesiana y distribuciones Dixon-Coles coherentes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

MAX_GOALS = 12
ARTIFACT_VERSION = 2


def _poisson_probabilities(rate: float, maximum: int = MAX_GOALS) -> np.ndarray:
    rate = max(float(rate), 1e-6)
    values = np.empty(maximum + 1, dtype=float)
    values[0] = np.exp(-rate)
    for goal in range(1, maximum + 1):
        values[goal] = values[goal - 1] * rate / goal
    # El ultimo bucket representa 12 o mas y conserva toda la masa sin ampliar el contrato 13x13.
    values[-1] += max(0.0, 1.0 - float(values.sum()))
    return values


def dixon_coles_matrix(rate_a: float, rate_b: float, rho: float = 0.0) -> tuple[np.ndarray, float]:
    """Devuelve P(goles A, goles B) y la masa Poisson truncada antes de normalizar."""
    matrix = np.outer(_poisson_probabilities(rate_a), _poisson_probabilities(rate_b))
    rho = float(np.clip(rho, -0.2, 0.2))
    matrix[0, 0] *= max(1e-8, 1.0 - rate_a * rate_b * rho)
    matrix[0, 1] *= max(1e-8, 1.0 + rate_a * rho)
    matrix[1, 0] *= max(1e-8, 1.0 + rate_b * rho)
    matrix[1, 1] *= max(1e-8, 1.0 - rho)
    retained_mass = float(matrix.sum())
    return matrix / retained_mass, retained_mass


def outcome_probabilities(matrix: np.ndarray) -> np.ndarray:
    return np.array([
        np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()
    ], dtype=float)


def align_score_matrix(matrix: np.ndarray, target: Iterable[float]) -> np.ndarray:
    """Conserva P(marcador|resultado) y reemplaza la masa 1-X-2 por la del DL."""
    source = np.asarray(matrix, dtype=float)
    target_array = np.asarray(tuple(target), dtype=float)
    target_array = target_array / target_array.sum()
    masks = (
        np.tri(source.shape[0], source.shape[1], -1, dtype=bool),
        np.eye(source.shape[0], source.shape[1], dtype=bool),
        np.triu(np.ones_like(source, dtype=bool), 1),
    )
    aligned = np.zeros_like(source)
    for probability, mask in zip(target_array, masks, strict=True):
        mass = float(source[mask].sum())
        if mass <= 0.0:
            raise ValueError("La distribucion estadistica no contiene uno de los resultados 1-X-2")
        aligned[mask] = source[mask] * probability / mass
    return aligned / aligned.sum()


@dataclass(frozen=True)
class CalibrationPosterior:
    temperatures: np.ndarray
    biases: np.ndarray

    def calibrate(self, probabilities: np.ndarray, posterior_draw: int | None = None) -> np.ndarray:
        probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0)
        logits = np.log(probabilities)
        if posterior_draw is None:
            temperature = float(np.mean(self.temperatures))
            bias = np.mean(self.biases, axis=0)
        else:
            index = int(posterior_draw) % len(self.temperatures)
            temperature = float(self.temperatures[index])
            bias = self.biases[index]
        adjusted = logits / max(temperature, 1e-3) + bias
        adjusted -= adjusted.max(axis=-1, keepdims=True)
        result = np.exp(adjusted)
        return result / result.sum(axis=-1, keepdims=True)

    @classmethod
    def identity(cls, draws: int = 64) -> "CalibrationPosterior":
        return cls(np.ones(draws), np.zeros((draws, 3)))


@dataclass(frozen=True)
class DixonColesPosterior:
    teams: tuple[str, ...]
    attack: np.ndarray
    defense: np.ndarray
    intercept: np.ndarray
    rho: np.ndarray
    world_cup_effect: np.ndarray

    def rates(self, team_a: str, team_b: str, posterior_draw: int | None = None) -> tuple[float, float, float]:
        lookup = {team: index for index, team in enumerate(self.teams)}
        if posterior_draw is None:
            attack = self.attack.mean(axis=0)
            defense = self.defense.mean(axis=0)
            intercept = float(self.intercept.mean())
            rho = float(self.rho.mean())
            tournament = float(self.world_cup_effect.mean())
        else:
            draw = int(posterior_draw) % len(self.intercept)
            attack, defense = self.attack[draw], self.defense[draw]
            intercept, rho = float(self.intercept[draw]), float(self.rho[draw])
            tournament = float(self.world_cup_effect[draw])
        a = lookup.get(team_a)
        b = lookup.get(team_b)
        attack_a = 0.0 if a is None else float(attack[a])
        attack_b = 0.0 if b is None else float(attack[b])
        defense_a = 0.0 if a is None else float(defense[a])
        defense_b = 0.0 if b is None else float(defense[b])
        return (
            float(np.exp(np.clip(intercept + tournament + attack_a - defense_b, -3.0, 3.0))),
            float(np.exp(np.clip(intercept + tournament + attack_b - defense_a, -3.0, 3.0))),
            rho,
        )

    @classmethod
    def neutral(cls, teams: Iterable[str], draws: int = 64) -> "DixonColesPosterior":
        names = tuple(sorted(set(teams)))
        return cls(
            teams=names,
            attack=np.zeros((draws, len(names))), defense=np.zeros((draws, len(names))),
            intercept=np.full(draws, np.log(1.25)), rho=np.zeros(draws),
            world_cup_effect=np.zeros(draws),
        )


def fit_bayesian_calibrator(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    draws: int = 64,
    advi_steps: int = 50_000,
    seed: int = 2026,
) -> tuple[CalibrationPosterior, dict[str, float]]:
    """Ajusta temperature scaling con sesgos por clase y priors regularizadores."""
    import pymc as pm
    from pymc.variational.callbacks import CheckParametersConvergence
    import pytensor.tensor as pt

    logits = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0))
    with pm.Model() as model:
        log_temperature = pm.Normal("log_temperature", mu=0.0, sigma=0.35)
        bias_raw = pm.Normal("bias_raw", mu=0.0, sigma=0.35, shape=3)
        bias = pm.Deterministic("bias", bias_raw - pt.mean(bias_raw))
        calibrated = pm.math.softmax(logits / pt.exp(log_temperature) + bias, axis=1)
        pm.Categorical("outcome", p=calibrated, observed=np.asarray(outcomes, dtype=np.int32))
        callback = CheckParametersConvergence(tolerance=1e-4)
        approximation = pm.fit(advi_steps, method="advi", callbacks=[callback], random_seed=seed, progressbar=False)
        trace = approximation.sample(draws=draws, random_seed=seed, return_inferencedata=True)
    temperatures = np.exp(trace.posterior["log_temperature"].values.reshape(-1))
    biases = trace.posterior["bias"].values.reshape(-1, 3)
    history = np.asarray(approximation.hist, dtype=float)
    diagnostics = {
        "iterations": float(len(history)),
        "final_elbo": float(history[-1]),
        "elbo_relative_change": float(abs(history[-1] - history[max(0, len(history) - 101)]) / max(abs(history[-1]), 1.0)),
    }
    return CalibrationPosterior(temperatures, biases), diagnostics


def fit_dixon_coles(
    frame,
    draws: int = 64,
    advi_steps: int = 50_000,
    seed: int = 2026,
) -> tuple[DixonColesPosterior, dict[str, float]]:
    """Ajusta fortalezas anuales jerarquicas y la correccion Dixon-Coles."""
    import pymc as pm
    import pytensor.tensor as pt
    from pymc.variational.callbacks import CheckParametersConvergence

    ordered = frame.sort_values("date").reset_index(drop=True)
    teams = tuple(sorted(set(ordered["home_team"]) | set(ordered["away_team"])))
    years = tuple(range(int(ordered["date"].dt.year.min()), int(ordered["date"].dt.year.max()) + 1))
    team_index = {team: index for index, team in enumerate(teams)}
    year_index = {year: index for index, year in enumerate(years)}
    home = ordered["home_team"].map(team_index).to_numpy(np.int32)
    away = ordered["away_team"].map(team_index).to_numpy(np.int32)
    period = ordered["date"].dt.year.map(year_index).to_numpy(np.int32)
    neutral = ordered["neutral"].astype(float).to_numpy()
    goals_a = ordered["home_score"].to_numpy(np.int32)
    goals_b = ordered["away_score"].to_numpy(np.int32)
    category_names = ("other", "friendly", "qualification", "continental", "world_cup")
    tournament = ordered["tournament"].str.lower()
    category = np.select(
        [
            tournament.str.contains("friendly", na=False),
            tournament.str.contains("qualification", na=False),
            tournament.str.contains("euro|copa america|african cup|asian cup|gold cup|nations league", regex=True, na=False),
            tournament.eq("fifa world cup"),
        ],
        [1, 2, 3, 4], default=0,
    ).astype(np.int32)
    coords = {
        "team": teams, "year": years, "match": np.arange(len(ordered)),
        "tournament_category": category_names,
    }
    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", mu=np.log(1.25), sigma=0.3)
        home_effect = pm.Normal("home_effect", mu=0.15, sigma=0.15)
        tournament_effect_raw = pm.Normal(
            "tournament_effect_raw", mu=0.0, sigma=0.15, dims="tournament_category"
        )
        tournament_effect = pm.Deterministic(
            "tournament_effect",
            tournament_effect_raw - pt.mean(tournament_effect_raw),
            dims="tournament_category",
        )
        sigma_attack = pm.HalfNormal("sigma_attack", sigma=0.15)
        sigma_defense = pm.HalfNormal("sigma_defense", sigma=0.15)
        attack_raw = pm.GaussianRandomWalk(
            "attack_raw", sigma=sigma_attack,
            init_dist=pm.Normal.dist(0.0, 0.35), dims=("team", "year"),
        )
        defense_raw = pm.GaussianRandomWalk(
            "defense_raw", sigma=sigma_defense,
            init_dist=pm.Normal.dist(0.0, 0.35), dims=("team", "year"),
        )
        attack = pm.Deterministic("attack", attack_raw - pt.mean(attack_raw, axis=0), dims=("team", "year"))
        defense = pm.Deterministic("defense", defense_raw - pt.mean(defense_raw, axis=0), dims=("team", "year"))
        rho_raw = pm.Normal("rho_raw", mu=0.0, sigma=0.5)
        rho = pm.Deterministic("rho", 0.2 * pt.tanh(rho_raw))
        log_a = intercept + attack[home, period] - defense[away, period]
        log_b = intercept + attack[away, period] - defense[home, period]
        log_a += home_effect * (1.0 - neutral) + tournament_effect[category]
        log_b += tournament_effect[category]
        rate_a, rate_b = pt.exp(log_a), pt.exp(log_b)
        pm.Poisson("goals_a", mu=rate_a, observed=goals_a, dims="match")
        pm.Poisson("goals_b", mu=rate_b, observed=goals_b, dims="match")
        tau = pt.ones_like(rate_a)
        tau = pt.switch((goals_a == 0) & (goals_b == 0), 1.0 - rate_a * rate_b * rho, tau)
        tau = pt.switch((goals_a == 0) & (goals_b == 1), 1.0 + rate_a * rho, tau)
        tau = pt.switch((goals_a == 1) & (goals_b == 0), 1.0 + rate_b * rho, tau)
        tau = pt.switch((goals_a == 1) & (goals_b == 1), 1.0 - rho, tau)
        pm.Potential("dixon_coles_adjustment", pt.sum(pt.log(pt.clip(tau, 1e-8, np.inf))))
        callback = CheckParametersConvergence(tolerance=1e-4)
        approximation = pm.fit(advi_steps, method="advi", callbacks=[callback], random_seed=seed, progressbar=False)
        trace = approximation.sample(draws=draws, random_seed=seed, return_inferencedata=True)
    posterior = trace.posterior
    attack_draws = posterior["attack"].values.reshape(-1, len(teams), len(years))[:, :, -1]
    defense_draws = posterior["defense"].values.reshape(-1, len(teams), len(years))[:, :, -1]
    result = DixonColesPosterior(
        teams=teams,
        attack=attack_draws,
        defense=defense_draws,
        intercept=posterior["intercept"].values.reshape(-1),
        rho=posterior["rho"].values.reshape(-1),
        world_cup_effect=posterior["tournament_effect"].values.reshape(-1, len(category_names))[:, 4],
    )
    history = np.asarray(approximation.hist, dtype=float)
    diagnostics = {
        "iterations": float(len(history)), "final_elbo": float(history[-1]),
        "elbo_relative_change": float(abs(history[-1] - history[max(0, len(history) - 101)]) / max(abs(history[-1]), 1.0)),
        "teams": float(len(teams)), "years": float(len(years)),
    }
    return result, diagnostics


def audit_dixon_coles_nuts(frame, seed: int = 2026) -> dict[str, float]:
    """Auditoria NUTS reducida de parametros globales sobre equipos de World Cup 2022."""
    import arviz as az
    import pymc as pm

    world_cup = frame.loc[
        frame["tournament"].str.fullmatch("FIFA World Cup", case=False)
        & frame["date"].between("2022-11-20", "2022-12-18")
    ]
    teams = set(world_cup["home_team"]) | set(world_cup["away_team"])
    audit = frame.loc[
        frame["date"].between("2020-01-01", "2022-11-19")
        & frame["home_team"].isin(teams) & frame["away_team"].isin(teams)
    ].copy()
    with pm.Model() as model:
        intercept = pm.Normal("intercept", np.log(1.25), 0.3)
        home_effect = pm.Normal("home_effect", 0.15, 0.15)
        rate_a = pm.math.exp(intercept + home_effect * (1.0 - audit["neutral"].astype(float).to_numpy()))
        rate_b = pm.math.exp(intercept)
        pm.Poisson("goals_a", rate_a, observed=audit["home_score"].to_numpy(np.int32))
        pm.Poisson("goals_b", rate_b, observed=audit["away_score"].to_numpy(np.int32))
        trace = pm.sample(
            draws=1_000, tune=1_000, chains=4, cores=4, random_seed=seed,
            target_accept=0.9, progressbar=False,
        )
    summary = az.summary(trace, var_names=["intercept", "home_effect"])
    return {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "matches": float(len(audit)),
    }
