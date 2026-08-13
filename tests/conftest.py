"""Shared fixtures.

The project has no package structure, so tests import `main` by path. Every
fixture that touches Keras is session-scoped: loading MNIST and the saved model
costs a couple of seconds and nothing here mutates them.
"""
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent / "neuralnetwork" / "MNSIT-NEURAL-NETWORK"
MODEL_PATH = PROJECT_DIR / "mnist_model.keras"

sys.path.insert(0, str(PROJECT_DIR))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full 10000-image evaluation (~10s)")


@pytest.fixture(scope="session")
def main_module():
    """The module under test. Skips the suite if Keras is not installed."""
    keras = pytest.importorskip("keras", reason="pip install -r requirements.txt")
    import main
    return main


@pytest.fixture(scope="session")
def model_path():
    if not MODEL_PATH.exists():
        pytest.skip(f"trained model missing at {MODEL_PATH}")
    return str(MODEL_PATH)


@pytest.fixture(scope="session")
def model(model_path):
    import keras
    return keras.saving.load_model(model_path)


@pytest.fixture(scope="session")
def mnist(main_module):
    """clean_data() output over the real MNIST download."""
    import keras
    return main_module.clean_data(keras.datasets.mnist.load_data())
