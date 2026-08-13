"""clean_data(): shapes, dtypes, and the binarization contract.

Binarization is the load-bearing preprocessing step. Everything downstream
assumes inputs are exactly 0 or 1, because that is what removes multiplication
from the 784-wide layer.
"""
import numpy as np


def test_shapes(mnist):
    X_train, Y_train, X_test, Y_test = mnist
    assert X_train.shape == (60000, 784)
    assert Y_train.shape == (60000, 10)
    assert X_test.shape == (10000, 784)
    assert Y_test.shape == (10000, 10)


def test_inputs_are_int8(mnist):
    X_train, _, X_test, _ = mnist
    assert X_train.dtype == np.int8
    assert X_test.dtype == np.int8


def test_inputs_are_strictly_binary(mnist):
    """Not 'mostly binary'. Any other value breaks the add-or-skip assumption."""
    X_train, _, X_test, _ = mnist
    assert set(np.unique(X_train).tolist()) == {0, 1}
    assert set(np.unique(X_test).tolist()) == {0, 1}


def test_labels_are_one_hot(mnist):
    _, Y_train, _, Y_test = mnist
    for Y in (Y_train, Y_test):
        assert np.array_equal(np.unique(Y), np.array([0.0, 1.0]))
        assert np.all(Y.sum(axis=1) == 1)


def test_binarization_threshold_is_128(main_module):
    """Threshold is 0.5 after dividing by 255, so raw byte 127 dies and 128 lives."""
    images = np.zeros((60000, 28, 28), dtype=np.uint8)
    images[0, 0, 0] = 127
    images[0, 0, 1] = 128
    images[0, 0, 2] = 255
    labels = np.zeros(60000, dtype=np.uint8)

    test_images = np.zeros((10000, 28, 28), dtype=np.uint8)
    test_labels = np.zeros(10000, dtype=np.uint8)

    X_train, _, _, _ = main_module.clean_data(
        ((images, labels), (test_images, test_labels))
    )
    assert X_train[0][0] == 0, "raw 127 is below the 0.5 threshold"
    assert X_train[0][1] == 1, "raw 128 is at or above the 0.5 threshold"
    assert X_train[0][2] == 1


def test_roughly_thirteen_percent_of_pixels_survive(mnist):
    """Sanity check on the sparsity the inner `if pixel == 1` guard exploits."""
    _, _, X_test, _ = mnist
    lit_fraction = X_test.sum() / X_test.size
    assert 0.10 < lit_fraction < 0.20
