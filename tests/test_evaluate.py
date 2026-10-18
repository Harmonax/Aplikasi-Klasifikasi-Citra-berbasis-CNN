import csv

import pytest
from PIL import Image

import evaluate_images
from evaluate_images import (
    CSV_COLUMNS,
    collect_images,
    confusion_matrix,
    evaluate,
    format_confusion_matrix,
    summarize,
    write_csv,
)
from model_utils import CLASS_NAMES


class FixedModel:
    """Model palsu yang selalu memprediksi 'cat'."""

    def predict(self, x, verbose=0):
        import numpy as np

        probs = np.full((1, 10), 0.01, dtype="float32")
        probs[0, 3] = 0.91  # cat
        return probs


@pytest.fixture
def image_dir(tmp_path):
    root = tmp_path / "test_images"
    for label, n in [("cat", 2), ("ship", 1)]:
        (root / label).mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (40, 40), color=(i * 50, 0, 0)).save(root / label / f"{i}.png")
    (root / "cat" / "catatan.txt").write_text("bukan gambar")
    (root / "cat" / "rusak.jpg").write_bytes(b"bukan jpeg")
    (root / "pesawat").mkdir()
    Image.new("RGB", (10, 10)).save(root / "pesawat" / "x.png")
    return root


def test_collect_images_skips_invalid_folder(image_dir):
    items, warnings = collect_images(image_dir)
    labels = sorted(label for _, label in items)
    assert labels == ["cat", "cat", "cat", "ship"]  # 2 png + rusak.jpg, txt diabaikan
    assert any("pesawat" in w for w in warnings)


def test_evaluate_and_summary(image_dir):
    rows, warnings = evaluate(FixedModel(), image_dir)
    assert len(rows) == 3  # rusak.jpg dilewati
    assert any("rusak.jpg" in w for w in warnings)
    assert all(r["prediksi"] == "cat" for r in rows)

    s = summarize(rows)
    assert s["total"] == 3 and s["benar"] == 2
    assert s["akurasi"] == pytest.approx(2 / 3)
    assert s["per_kelas"]["cat"]["akurasi"] == 1.0
    assert s["per_kelas"]["ship"]["akurasi"] == 0.0


def test_confusion_matrix(image_dir):
    rows, _ = evaluate(FixedModel(), image_dir)
    m = confusion_matrix(rows)
    cat, ship = CLASS_NAMES.index("cat"), CLASS_NAMES.index("ship")
    assert len(m) == 10 and all(len(r) == 10 for r in m)
    assert m[cat][cat] == 2  # 2 gambar cat diprediksi cat
    assert m[ship][cat] == 1  # 1 gambar ship salah diprediksi cat
    assert sum(map(sum, m)) == len(rows)

    text = format_confusion_matrix(m)
    body = [line.split()[0] for line in text.splitlines()[1:]]
    assert body == ["cat", "ship"]  # hanya kelas yang punya gambar


def test_write_csv(image_dir, tmp_path):
    rows, _ = evaluate(FixedModel(), image_dir)
    out = tmp_path / "hasil.csv"
    write_csv(rows, out)
    with open(out, encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    assert list(data[0].keys()) == CSV_COLUMNS
    assert len(data) == 3


def test_main_with_dummy_model(image_dir, tmp_path, dummy_model_path, capsys):
    out = tmp_path / "hasil.csv"
    code = evaluate_images.main(
        ["--dir", str(image_dir), "--out", str(out), "--model", str(dummy_model_path)]
    )
    assert code == 0
    assert out.is_file()
    assert "Akurasi" in capsys.readouterr().out


def test_main_empty_dir(tmp_path, dummy_model_path):
    (tmp_path / "cat").mkdir()
    code = evaluate_images.main(["--dir", str(tmp_path), "--model", str(dummy_model_path),
                                 "--out", str(tmp_path / "o.csv")])
    assert code == 1
