"""The integer forward pass: architecture assumptions, output contract, accuracy.

forward() prints instead of returning, so these tests parse stdout. That is the
function's actual interface today.
"""
import re

import numpy as np
import pytest


def parse_accuracy(stdout):
    """Pull (correct, total) out of the final 'Accuracy: a / b = c%' line."""
    match = re.search(r"Accuracy: (\d+) / (\d+) = ([\d.]+)%", stdout)
    assert match, f"no accuracy line in output:\n{stdout[-500:]}"
    return int(match.group(1)), int(match.group(2)), float(match.group(3))


class TestArchitectureAssumptions:
    """forward() hardcodes layer indices 0 and 2. Guard that contract."""

    def test_dense_layers_at_indices_zero_and_two(self, model):
        import keras
        assert isinstance(model.layers[0], keras.layers.Dense)
        assert isinstance(model.layers[2], keras.layers.Dense)

    def test_activations_are_separate_layers(self, model):
        """Collapsing these into Dense(activation=...) shifts the indices."""
        import keras
        assert isinstance(model.layers[1], keras.layers.Activation)
        assert isinstance(model.layers[3], keras.layers.Activation)

    def test_weight_shapes(self, model):
        kernel1, bias1 = model.layers[0].get_weights()
        kernel2, bias2 = model.layers[2].get_weights()
        assert kernel1.shape == (784, 10)
        assert bias1.shape == (10,)
        assert kernel2.shape == (10, 10)
        assert bias2.shape == (10,)

    def test_get_weights_returns_exactly_two_arrays(self, model):
        """Regression guard: forward() once indexed get_weights()[50] here."""
        assert len(model.layers[0].get_weights()) == 2
        assert len(model.layers[2].get_weights()) == 2

    def test_parameter_count(self, model):
        assert model.count_params() == 7960


class TestOutputContract:
    def test_prints_accuracy_line(self, main_module, model_path, capsys):
        main_module.forward(model_path, 100)
        correct, total, percent = parse_accuracy(capsys.readouterr().out)
        assert total == 100
        assert 0 <= correct <= total
        assert percent == pytest.approx(correct / total * 100, abs=0.01)

    def test_progress_lines_every_hundred(self, main_module, model_path, capsys):
        main_module.forward(model_path, 250)
        lines = capsys.readouterr().out.strip().splitlines()
        assert "100" in lines
        assert "200" in lines

    def test_final_progress_line_is_skipped(self, main_module, model_path, capsys):
        """The loop breaks before its last print, so 200 never appears at n=200."""
        main_module.forward(model_path, 200)
        lines = capsys.readouterr().out.strip().splitlines()
        assert "100" in lines
        assert "200" not in lines

    def test_iterations_below_one_still_classifies_one(self, main_module, model_path, capsys):
        """total is incremented and checked after the first image."""
        main_module.forward(model_path, 0)
        _, total, _ = parse_accuracy(capsys.readouterr().out)
        assert total == 1

    def test_iterations_capped_by_test_set(self, main_module, model_path, capsys):
        main_module.forward(model_path, 999999)
        _, total, _ = parse_accuracy(capsys.readouterr().out)
        assert total == 10000


class TestAccuracy:
    def test_subset_is_in_a_sane_band(self, main_module, model_path, capsys):
        main_module.forward(model_path, 500)
        _, _, percent = parse_accuracy(capsys.readouterr().out)
        assert 75.0 < percent < 95.0, "integer pass should be well above chance"

    def test_beats_chance_by_a_wide_margin(self, main_module, model_path, capsys):
        main_module.forward(model_path, 300)
        _, _, percent = parse_accuracy(capsys.readouterr().out)
        assert percent > 50.0, "10% is chance on 10 classes; a collapse means overflow"

    @pytest.mark.slow
    def test_full_test_set_regression_lock(self, main_module, model_path, capsys):
        """Exact figure for the shipped weights. Deterministic: no RNG anywhere.

        If you retrain, this number moves and the docs need updating with it.
        """
        main_module.forward(model_path, 10000)
        correct, total, percent = parse_accuracy(capsys.readouterr().out)
        assert (correct, total) == (8544, 10000)
        assert percent == 85.44

    @pytest.mark.slow
    def test_integer_pass_stays_below_float_model(self, main_module, model, mnist, model_path, capsys):
        """Quantization costs accuracy. Measured gap is just under 7 points."""
        _, _, X_test, Y_test = mnist
        float_accuracy = model.evaluate(X_test, Y_test, verbose=0)[1]

        main_module.forward(model_path, 10000)
        _, _, integer_percent = parse_accuracy(capsys.readouterr().out)

        assert integer_percent < float_accuracy * 100
        assert float_accuracy * 100 - integer_percent < 15.0


class TestKnownEdgeCases:
    def test_ties_produce_multiple_winners(self):
        """np.where lights every maximum, so a tie can never equal a one-hot label."""
        output = np.array([5, 5, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16)
        prediction = np.where(output == np.max(output), 1, 0).astype(np.int16)
        assert prediction.sum() == 2

        label = np.zeros(10)
        label[0] = 1
        assert not np.array_equal(prediction, label), "tie scores as wrong even when right"

    def test_relu_clamps_negatives_to_zero(self):
        hidden = np.array([13, -6, 47, -128, 0], dtype=np.int8)
        activated = np.maximum(0, hidden).astype(np.int8)
        assert activated.tolist() == [13, 0, 47, 0, 0]
