"""Arquitecturas Keras exigidas por la rubrica."""

from __future__ import annotations

from typing import Literal


def _tensorflow():
    import tensorflow as tf

    return tf


def make_optimizer(name: Literal["adam", "sgd"] = "adam"):
    tf = _tensorflow()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=0.001)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    raise ValueError(f"Optimizador desconocido: {name}")


def _heads(embedding):
    tf = _tensorflow()
    result = tf.keras.layers.Dense(3, activation="softmax", name="result")(embedding)
    goals_a = tf.keras.layers.Dense(1, activation="softplus", name="goals_a")(embedding)
    goals_b = tf.keras.layers.Dense(1, activation="softplus", name="goals_b")(embedding)
    return result, goals_a, goals_b


def compile_multitask(model, optimizer="adam"):
    tf = _tensorflow()
    model.compile(
        optimizer=make_optimizer(optimizer) if isinstance(optimizer, str) else optimizer,
        loss={
            "result": "sparse_categorical_crossentropy",
            "goals_a": tf.keras.losses.Poisson(),
            "goals_b": tf.keras.losses.Poisson(),
        },
        loss_weights={"result": 1.0, "goals_a": 0.25, "goals_b": 0.25},
        metrics={"result": ["accuracy"], "goals_a": ["mae"], "goals_b": ["mae"]},
    )
    return model


def build_mlp(input_dim: int, optimizer: Literal["adam", "sgd"] = "adam"):
    tf = _tensorflow()
    inputs = tf.keras.Input(shape=(input_dim,), name="static")
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    model = tf.keras.Model(inputs=inputs, outputs=_heads(x), name=f"mlp_{optimizer}")
    return compile_multitask(model, optimizer)


def build_recurrent(
    static_dim: int,
    sequence_dim: int,
    cell: Literal["lstm", "gru"] = "lstm",
    lookback: int = 10,
):
    tf = _tensorflow()
    layer_class = tf.keras.layers.LSTM if cell == "lstm" else tf.keras.layers.GRU
    sequence_input = tf.keras.Input(shape=(lookback, sequence_dim), name="encoder_input")
    encoded = layer_class(64, return_sequences=True)(sequence_input)
    encoded = tf.keras.layers.Dropout(0.25)(encoded)
    encoded = layer_class(32)(encoded)
    encoded = tf.keras.layers.Dropout(0.25)(encoded)
    encoder = tf.keras.Model(sequence_input, encoded, name=f"shared_{cell}_encoder")

    static = tf.keras.Input(shape=(static_dim,), name="static")
    team_a = tf.keras.Input(shape=(lookback, sequence_dim), name="sequence_a")
    team_b = tf.keras.Input(shape=(lookback, sequence_dim), name="sequence_b")
    merged = tf.keras.layers.Concatenate()([static, encoder(team_a), encoder(team_b)])
    merged = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(merged)
    merged = tf.keras.layers.Dropout(0.25)(merged)
    merged = tf.keras.layers.Dense(32, activation="relu")(merged)
    model = tf.keras.Model(
        inputs={"static": static, "sequence_a": team_a, "sequence_b": team_b},
        outputs=_heads(merged),
        name=f"{cell}_trajectory",
    )
    return compile_multitask(model, "adam")
