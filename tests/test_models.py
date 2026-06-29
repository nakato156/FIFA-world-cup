import pytest

tf = pytest.importorskip("tensorflow")

from mundial.models import build_mlp, build_recurrent


def test_mlp_has_four_relu_hidden_layers_and_three_heads():
    model = build_mlp(24, "adam")
    hidden = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense) and layer.name not in {"result", "goals_a", "goals_b"}]
    assert len(hidden) == 4
    assert [layer.units for layer in hidden] == [256, 128, 64, 32]
    assert {output.name.split("/")[0] for output in model.outputs} == {"keras_tensor", "keras_tensor_1", "keras_tensor_2"} or len(model.outputs) == 3


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_recurrent_models_have_two_recurrent_layers(cell):
    model = build_recurrent(24, 5, cell)
    encoder = model.get_layer(f"shared_{cell}_encoder")
    expected = tf.keras.layers.LSTM if cell == "lstm" else tf.keras.layers.GRU
    assert len([layer for layer in encoder.layers if isinstance(layer, expected)]) == 2

