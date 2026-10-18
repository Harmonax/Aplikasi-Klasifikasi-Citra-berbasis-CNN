import numpy as np
import pytest
from PIL import Image

from model_utils import (
    CLASS_NAMES,
    IMG_SIZE,
    build_prob_df,
    load_model,
    predict,
    preprocess_image,
    resolve_model_path,
)


def test_class_names_order():
    assert CLASS_NAMES == [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    assert IMG_SIZE == (32, 32)


def test_preprocess_shape_dtype_range():
    img = Image.new("RGB", (100, 80), color=(255, 128, 0))
    x = preprocess_image(img)
    assert x.shape == (1, 32, 32, 3)
    assert x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0
    np.testing.assert_allclose(x[0, 0, 0], [1.0, 128 / 255, 0.0], atol=1e-6)


@pytest.mark.parametrize("mode", ["RGBA", "L", "P", "CMYK"])
def test_preprocess_converts_modes_to_rgb(mode):
    img = Image.new(mode, (50, 50))
    assert preprocess_image(img).shape == (1, 32, 32, 3)


def test_preprocess_non_square_image():
    img = Image.new("RGB", (640, 200))
    assert preprocess_image(img).shape == (1, 32, 32, 3)


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "tidak_ada.keras")


def test_resolve_model_path_env(monkeypatch, tmp_path):
    target = tmp_path / "m.keras"
    monkeypatch.setenv("MODEL_PATH", str(target))
    assert resolve_model_path() == target
    assert resolve_model_path("lain.keras").name == "lain.keras"


def test_load_dummy_model(dummy_model):
    assert dummy_model.input_shape == (None, 32, 32, 3)
    assert dummy_model.output_shape == (None, 10)


def test_predict_returns_sorted_probs(dummy_model):
    img = Image.new("RGB", (64, 64), color=(10, 200, 30))
    results = predict(dummy_model, img)
    assert len(results) == 10
    assert {name for name, _ in results} == set(CLASS_NAMES)
    probs = [p for _, p in results]
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) == pytest.approx(1.0, abs=1e-4)


def test_build_prob_df():
    results = [(name, i / 45) for i, name in enumerate(CLASS_NAMES)]
    df = build_prob_df(results)
    assert list(df.columns) == ["kelas", "probabilitas"]
    assert len(df) == 10
    assert df["probabilitas"].is_monotonic_decreasing
    assert df.iloc[0]["kelas"] == "truck"
