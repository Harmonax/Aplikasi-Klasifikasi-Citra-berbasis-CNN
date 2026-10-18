from streamlit.testing.v1 import AppTest

from conftest import PROJECT_ROOT

APP_FILE = str(PROJECT_ROOT / "app.py")


def test_app_runs_with_model(monkeypatch, dummy_model_path):
    monkeypatch.setenv("MODEL_PATH", str(dummy_model_path))
    at = AppTest.from_file(APP_FILE, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Klasifikasi Gambar CIFAR-10"
    assert len(at.error) == 0
    assert len(at.info) == 1  # belum ada gambar


def test_app_shows_error_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "tidak_ada.keras"))
    at = AppTest.from_file(APP_FILE, default_timeout=60).run()
    assert not at.exception
    assert len(at.error) == 1
    assert "tidak ditemukan" in at.error[0].value


def _fake_upload(monkeypatch, data: bytes):
    """AppTest belum bisa upload file, jadi st.file_uploader diganti agar mengembalikan bytes."""
    import io

    import streamlit as st

    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: io.BytesIO(data))


def test_app_predicts_uploaded_image(monkeypatch, dummy_model_path):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (120, 90), color=(0, 120, 255, 255)).save(buf, format="PNG")
    monkeypatch.setenv("MODEL_PATH", str(dummy_model_path))
    _fake_upload(monkeypatch, buf.getvalue())

    at = AppTest.from_file(APP_FILE, default_timeout=60).run()
    assert not at.exception
    assert len(at.error) == 0
    assert [m.label for m in at.metric] == ["Prediksi", "Confidence"]
    assert at.metric[1].value.endswith("%")
    assert len(at.dataframe) == 1


def test_app_rejects_invalid_image(monkeypatch, dummy_model_path):
    monkeypatch.setenv("MODEL_PATH", str(dummy_model_path))
    _fake_upload(monkeypatch, b"ini bukan gambar")

    at = AppTest.from_file(APP_FILE, default_timeout=60).run()
    assert not at.exception
    assert "bukan gambar yang valid" in at.error[0].value
