"""Entrenamiento, comparacion y exportacion reproducible de modelos."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from mundial.config import ARTIFACTS_DIR, PROCESSED_DIR
from mundial.data import STATIC_FEATURES
from mundial.models import build_mlp, build_recurrent


def set_seeds(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass


def load_training_data(processed_dir: Path = PROCESSED_DIR):
    frame = pd.read_parquet(processed_dir / "matches.parquet")
    sequences = np.load(processed_dir / "sequences.npz")
    return frame, sequences["team_a"].astype(np.float32), sequences["team_b"].astype(np.float32)


def _targets(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "result": frame["result"].to_numpy(dtype=np.int32),
        "goals_a": frame["home_score"].to_numpy(dtype=np.float32),
        "goals_b": frame["away_score"].to_numpy(dtype=np.float32),
    }


def _inputs(kind: str, static: np.ndarray, seq_a: np.ndarray, seq_b: np.ndarray):
    if kind.startswith("mlp"):
        return static
    return {"static": static, "sequence_a": seq_a, "sequence_b": seq_b}


def _evaluate(model, inputs, frame: pd.DataFrame) -> dict[str, object]:
    probabilities, goals_a, goals_b = model.predict(inputs, verbose=0)
    truth = frame["result"].to_numpy()
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(3)[truth]
    return {
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "accuracy": float(accuracy_score(truth, prediction)),
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1))),
        "mae_goals_a": float(mean_absolute_error(frame["home_score"], goals_a.ravel())),
        "mae_goals_b": float(mean_absolute_error(frame["away_score"], goals_b.ravel())),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
    }


def _callbacks():
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_result_accuracy", patience=7, mode="max", restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def _team_inference_state(frame: pd.DataFrame, seq_a: np.ndarray, seq_b: np.ndarray):
    states: dict[str, dict[str, float]] = {}
    sequences: dict[str, np.ndarray] = {}
    ordered = frame.sort_values("date")
    for index, row in ordered.iterrows():
        for suffix, team_column, sequence_source in (
            ("a", "home_team", seq_a), ("b", "away_team", seq_b),
        ):
            team = row[team_column]
            states[team] = {
                "rank": float(row[f"rank_{suffix}"]),
                "goals_for_last10": float(row[f"goals_for_last10_{suffix}"]),
                "goals_against_last10": float(row[f"goals_against_last10_{suffix}"]),
                "players_imputed": float(row[f"players_imputed_{suffix}"]),
                **{attribute: float(row[f"{attribute}_{suffix}"]) for attribute in ("overall", "pace", "shooting", "defending", "physical")},
            }
            sequences[team] = sequence_source[index]
    numeric = pd.DataFrame(states.values()).median(numeric_only=True).to_dict()
    return states, sequences, {key: float(value) for key, value in numeric.items()}


def train_all(
    processed_dir: Path = PROCESSED_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    max_epochs_mlp: int = 60,
    max_epochs_recurrent: int = 40,
) -> dict[str, object]:
    set_seeds()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    frame, seq_a, seq_b = load_training_data(processed_dir)
    train_mask = frame["split"].eq("train").to_numpy()
    validation_mask = frame["split"].eq("validation").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("Las particiones train, validation y test deben contener partidos")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    raw = frame[list(STATIC_FEATURES)].to_numpy(dtype=np.float32)
    train_imputed = imputer.fit_transform(raw[train_mask])
    scaler.fit(train_imputed)
    static = scaler.transform(imputer.transform(raw)).astype(np.float32)
    candidates = {
        "mlp_adam": lambda: build_mlp(static.shape[1], "adam"),
        "mlp_sgd": lambda: build_mlp(static.shape[1], "sgd"),
        "lstm": lambda: build_recurrent(static.shape[1], seq_a.shape[2], "lstm"),
        "gru": lambda: build_recurrent(static.shape[1], seq_a.shape[2], "gru"),
    }
    report: dict[str, object] = {}
    trained = {}
    for name, factory in candidates.items():
        model = factory()
        epochs = max_epochs_mlp if name.startswith("mlp") else max_epochs_recurrent
        start = time.perf_counter()
        history = model.fit(
            _inputs(name, static[train_mask], seq_a[train_mask], seq_b[train_mask]),
            _targets(frame.loc[train_mask]),
            validation_data=(
                _inputs(name, static[validation_mask], seq_a[validation_mask], seq_b[validation_mask]),
                _targets(frame.loc[validation_mask]),
            ),
            epochs=epochs,
            batch_size=256,
            callbacks=_callbacks(),
            verbose=2,
        )
        elapsed = time.perf_counter() - start
        metrics = _evaluate(model, _inputs(name, static[test_mask], seq_a[test_mask], seq_b[test_mask]), frame.loc[test_mask])
        metrics["training_seconds"] = elapsed
        report[name] = metrics
        trained[name] = model
        (artifacts_dir / f"history_{name}.json").write_text(
            json.dumps({key: [float(value) for value in values] for key, values in history.history.items()}, indent=2),
            encoding="utf-8",
        )
    selected = max(report, key=lambda name: report[name]["macro_f1"])
    trained[selected].save(artifacts_dir / "selected_model.keras")
    states, latest_sequences, default_state = _team_inference_state(frame, seq_a, seq_b)
    joblib.dump(
        {
            "imputer": imputer, "scaler": scaler, "team_states": states,
            "team_sequences": latest_sequences, "default_state": default_state,
            "model_type": selected, "static_features": list(STATIC_FEATURES),
        },
        artifacts_dir / "inference_bundle.joblib",
    )
    summary = {"selected_model": selected, "models": report, "splits": frame["split"].value_counts().to_dict()}
    (artifacts_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

