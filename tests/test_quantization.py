"""Fixed-point conversion, overflow semantics, and the int8 accumulator.

Several assertions here lock in behavior that looks like a bug and is not.
Read docs/explanation-integer-inference.md before "fixing" anything they catch.
"""
import numpy as np
import pytest


def to_fixed(float_value, bits_past_radix=2):
    """Copy of the nested helper in forward(), which is not importable."""
    a = float_value * (2 ** bits_past_radix)
    b = int(round(a))
    if a < 0:
        b = ~(abs(b)) + 1
    return b


class TestToFixed:
    def test_scale_is_two_to_the_bits(self):
        assert to_fixed(1.0, 2) == 4
        assert to_fixed(1.0, 3) == 8
        assert to_fixed(0.25, 2) == 1

    def test_negatives_round_trip(self):
        """The ~abs(b)+1 branch is longhand two's complement, and a no-op here."""
        for value in (-0.25, -1.0, -1.6448, -7.75):
            assert to_fixed(value, 2) == int(round(value * 4))

    def test_resolution_halves_per_added_bit(self):
        weight = 0.3737
        errors = []
        for bits in (1, 2, 3, 4):
            reconstructed = to_fixed(weight, bits) / (2 ** bits)
            errors.append(abs(weight - reconstructed))
        for finer, coarser in zip(errors[1:], errors[:-1]):
            assert finer <= coarser

    def test_int8_range_at_default_bits(self):
        """2 fractional bits in an int8 covers -32.0 to +31.75."""
        assert to_fixed(31.75, 2) == 127
        assert to_fixed(-32.0, 2) == -128


class TestCheckOverflow:
    def test_detects_out_of_range(self, main_module, capsys):
        main_module.check_overflow(128, 8)
        assert "overflow detected" in capsys.readouterr().out

        main_module.check_overflow(-129, 8)
        assert "overflow detected" in capsys.readouterr().out

    def test_silent_inside_range(self, main_module, capsys):
        for value in (127, -128, 0):
            main_module.check_overflow(value, 8)
        assert capsys.readouterr().out == ""

    def test_sixteen_bit_bounds(self, main_module, capsys):
        main_module.check_overflow(32767, 16)
        assert capsys.readouterr().out == ""
        main_module.check_overflow(32768, 16)
        assert "overflow detected" in capsys.readouterr().out

    def test_blind_to_an_already_wrapped_int8(self, main_module, capsys):
        """KNOWN LIMITATION, asserted so it cannot regress silently.

        forward() passes a NumPy int8 that has already wrapped, so the value is
        inside the tested range by construction and this never fires. See
        docs/howto-tune-fixed-point.md#making-overflow-detection-actually-work.
        """
        with np.errstate(over="ignore"):
            wrapped = np.int8(100) + np.int8(100)
        assert wrapped == -56, "int8 addition wraps rather than promoting"

        main_module.check_overflow(wrapped, 8)
        assert capsys.readouterr().out == "", "the wrap is invisible to check_overflow"


class TestAccumulatorWidth:
    def test_int8_accumulation_wraps_by_design(self):
        """Models an 8-bit redstone adder with no carry-out. Do not widen this."""
        weights = np.array([100, 100], dtype=np.int8)
        acc = 0
        with np.errstate(over="ignore"):
            for w in weights:
                acc = acc + w
        assert acc.dtype == np.int8
        assert acc == -56

    def test_overflow_raises_under_errstate(self):
        """The documented way to make the wrap visible."""
        with pytest.raises(FloatingPointError):
            with np.errstate(over="raise"):
                np.int8(100) + np.int8(100)

    def test_int16_holds_the_output_layer(self):
        """int8 x int16 promotes to int16, which is why layer 2 does not wrap."""
        product = np.int8(127) * np.int16(127)
        assert product.dtype == np.int16
        assert product == 16129


class TestShippedWeights:
    def test_no_saturation_at_default_bits(self, model):
        """Weight saturation is not the binding constraint on this model."""
        for layer_index in (0, 2):
            kernel = model.layers[layer_index].get_weights()[0]
            scaled = np.round(kernel * 4)
            assert np.all((scaled <= 127) & (scaled >= -128))

    def test_saturation_begins_at_seven_bits(self, model):
        all_weights = np.concatenate([
            model.layers[0].get_weights()[0].ravel(),
            model.layers[2].get_weights()[0].ravel(),
        ])
        assert np.max(np.abs(all_weights)) < 2.0

        saturating_at_6 = np.sum(np.abs(np.round(all_weights * 64)) > 127)
        saturating_at_7 = np.sum(np.abs(np.round(all_weights * 128)) > 127)
        assert saturating_at_6 == 0
        assert saturating_at_7 > 0
