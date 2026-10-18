import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def dummy_model_path(tmp_path_factory):
    """Model kecil dengan input/output sama seperti model asli, disimpan sebagai .keras."""
    import keras
    from keras import layers

    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Flatten()(inputs)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    path = tmp_path_factory.mktemp("model") / "dummy.keras"
    model.save(path)
    return path


@pytest.fixture(scope="session")
def dummy_model(dummy_model_path):
    from model_utils import load_model

    return load_model(dummy_model_path)
