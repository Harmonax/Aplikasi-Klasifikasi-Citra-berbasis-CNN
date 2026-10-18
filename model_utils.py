"""Utilitas model CNN CIFAR-10: preprocessing, memuat model, dan prediksi.

Preprocessing di sini harus sama persis dengan saat training di Colab:
gambar RGB 32x32, ``astype('float32') / 255.0``.

Pemakaian CLI:
    python model_utils.py path/ke/gambar.jpg [--model path/ke/model.keras]
"""

from __future__ import annotations

import os
from pathlib import Path

# Kurangi log TensorFlow yang tidak perlu (harus di-set sebelum import tensorflow/keras).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageOps

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
IMG_SIZE = (32, 32)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "best_cifar10_tuned.keras"


def resolve_model_path(path: str | os.PathLike | None = None) -> Path:
    """Urutan prioritas: argumen ``path`` -> env var ``MODEL_PATH`` -> default."""
    if path is None:
        path = os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH
    return Path(path)


def to_model_view(pil_image: Image.Image) -> Image.Image:
    """Gambar RGB 32x32 persis seperti yang 'dilihat' model (sebelum normalisasi)."""
    img = ImageOps.exif_transpose(pil_image)  # perbaiki rotasi foto HP
    img = img.convert("RGB")  # RGBA/L/P -> 3 kanal
    return img.resize(IMG_SIZE, Image.Resampling.LANCZOS)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Ubah gambar PIL menjadi array (1, 32, 32, 3) float32 dengan nilai 0-1."""
    arr = np.asarray(to_model_view(pil_image)).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def load_model(path: str | os.PathLike | None = None):
    """Memuat model ``.keras``. Melempar FileNotFoundError jika file tidak ada."""
    model_path = resolve_model_path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {model_path}. "
            "Taruh file .keras di folder aplikasi atau set env var MODEL_PATH."
        )
    import keras  # import lambat supaya modul ini ringan saat hanya preprocessing

    return keras.saving.load_model(model_path)


def predict(model, pil_image: Image.Image) -> list[tuple[str, float]]:
    """Prediksi satu gambar. Hasil: [(kelas, probabilitas), ...] terurut menurun."""
    x = preprocess_image(pil_image)
    probs = model.predict(x, verbose=0)[0]  # output sudah softmax
    results = [(name, float(p)) for name, p in zip(CLASS_NAMES, probs)]
    return sorted(results, key=lambda item: item[1], reverse=True)


def build_prob_df(results: list[tuple[str, float]]):
    """DataFrame untuk chart/tabel: kolom ``kelas`` dan ``probabilitas``, terurut menurun."""
    import pandas as pd

    df = pd.DataFrame(results, columns=["kelas", "probabilitas"])
    return df.sort_values("probabilitas", ascending=False, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Prediksi kelas CIFAR-10 untuk satu gambar.")
    parser.add_argument("image", help="Path file gambar")
    parser.add_argument("--model", default=None, help="Path model .keras (opsional)")
    args = parser.parse_args(argv)

    model = load_model(args.model)
    with Image.open(args.image) as img:
        results = predict(model, img)

    print(f"Gambar : {args.image}")
    print(f"Prediksi: {results[0][0]} ({results[0][1]:.2%})\n")
    for name, prob in results:
        print(f"  {name:<11} {prob:7.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
