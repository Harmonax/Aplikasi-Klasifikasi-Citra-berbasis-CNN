"""Uji model CNN dengan gambar baru di luar dataset CIFAR-10.

Susun gambar per kelas (nama subfolder = label sebenarnya):

    test_images/
        cat/kucing1.jpg
        ship/kapal1.png
        ...

Jalankan:
    python evaluate_images.py --dir test_images --out hasil_uji.csv [--model path.keras]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from model_utils import CLASS_NAMES, IMAGE_EXTENSIONS, load_model, predict

CSV_COLUMNS = ["file", "label_asli", "prediksi", "confidence", "benar"]


def collect_images(root: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    """Kumpulkan (path gambar, label) dari subfolder kelas. Return juga daftar peringatan."""
    items: list[tuple[Path, str]] = []
    warnings: list[str] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        label = sub.name.lower()
        if label not in CLASS_NAMES:
            warnings.append(f"Folder '{sub.name}' bukan kelas CIFAR-10, dilewati.")
            continue
        for f in sorted(sub.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((f, label))
    return items, warnings


def evaluate(model, root: Path) -> tuple[list[dict], list[str]]:
    """Prediksi semua gambar. Return (baris hasil, peringatan)."""
    items, warnings = collect_images(root)
    rows: list[dict] = []
    for path, label in items:
        try:
            with Image.open(path) as img:
                pred, conf = predict(model, img)[0]
        except (UnidentifiedImageError, OSError):
            warnings.append(f"File '{path}' tidak bisa dibaca sebagai gambar, dilewati.")
            continue
        rows.append(
            {
                "file": path.relative_to(root).as_posix(),
                "label_asli": label,
                "prediksi": pred,
                "confidence": round(conf, 4),
                "benar": pred == label,
            }
        )
    return rows, warnings


def summarize(rows: list[dict]) -> dict:
    """Akurasi total dan per kelas."""
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [benar, total]
    for r in rows:
        per_class[r["label_asli"]][0] += int(r["benar"])
        per_class[r["label_asli"]][1] += 1
    total = len(rows)
    correct = sum(int(r["benar"]) for r in rows)
    return {
        "total": total,
        "benar": correct,
        "akurasi": correct / total if total else 0.0,
        "per_kelas": {
            k: {"benar": c, "total": t, "akurasi": c / t}
            for k, (c, t) in sorted(per_class.items(), key=lambda kv: CLASS_NAMES.index(kv[0]))
        },
    }


def confusion_matrix(rows: list[dict]) -> list[list[int]]:
    """Matriks 10x10 urut CLASS_NAMES: baris = label asli, kolom = prediksi."""
    idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    matrix = [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES]
    for r in rows:
        matrix[idx[r["label_asli"]]][idx[r["prediksi"]]] += 1
    return matrix


def format_confusion_matrix(matrix: list[list[int]]) -> str:
    """Tabel teks; hanya baris kelas yang punya gambar uji yang ditampilkan."""
    short = [name[:5] for name in CLASS_NAMES]
    lines = ["asli \\ pred " + " ".join(f"{s:>5}" for s in short)]
    for name, counts in zip(CLASS_NAMES, matrix):
        if sum(counts):
            cells = " ".join(f"{c if c else '.':>5}" for c in counts)
            lines.append(f"{name:<11} {cells}")
    return "\n".join(lines)


def write_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_report(rows: list[dict], summary: dict) -> None:
    print(f"\nTotal gambar : {summary['total']}")
    print(f"Benar        : {summary['benar']}")
    print(f"Akurasi      : {summary['akurasi']:.2%}\n")
    print("Akurasi per kelas:")
    for name, s in summary["per_kelas"].items():
        print(f"  {name:<11} {s['benar']:>3}/{s['total']:<3} {s['akurasi']:7.2%}")
    print("\nConfusion matrix (baris = label asli, kolom = prediksi):")
    print(format_confusion_matrix(confusion_matrix(rows)))
    wrong = [r for r in rows if not r["benar"]]
    if wrong:
        print("\nPrediksi salah:")
        for r in wrong:
            print(f"  {r['file']}: asli={r['label_asli']}, prediksi={r['prediksi']} "
                  f"({r['confidence']:.2%})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluasi model dengan gambar baru per kelas.")
    parser.add_argument("--dir", default="test_images", help="Folder berisi subfolder per kelas")
    parser.add_argument("--out", default="hasil_uji.csv", help="Path file CSV output")
    parser.add_argument("--model", default=None, help="Path model .keras (opsional)")
    args = parser.parse_args(argv)

    root = Path(args.dir)
    if not root.is_dir():
        print(f"Folder '{root}' tidak ditemukan.", file=sys.stderr)
        return 1

    model = load_model(args.model)
    rows, warnings = evaluate(model, root)
    for w in warnings:
        print(f"[peringatan] {w}")

    if not rows:
        print(f"Tidak ada gambar yang bisa diuji di '{root}'. "
              "Isi subfolder per kelas, misalnya test_images/cat/.")
        return 1

    write_csv(rows, Path(args.out))
    print_report(rows, summarize(rows))
    print(f"\nHasil lengkap disimpan ke: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
