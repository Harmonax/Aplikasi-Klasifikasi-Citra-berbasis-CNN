---
title: Klasifikasi Citra Berbasis CNN dari Data CIFAR-10
emoji: 🖼️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Klasifikasi Citra Berbasis CNN dari Data CIFAR-10

Aplikasi web Streamlit untuk mengklasifikasikan gambar ke 10 kelas CIFAR-10
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
menggunakan model CNN yang dilatih dan dioptimasi di Google Colab.

| File | Fungsi |
|---|---|
| `app.py` | Aplikasi web Streamlit (upload gambar → prediksi + probabilitas tiap kelas) |
| `model_utils.py` | Preprocessing, memuat model, prediksi (bisa juga dipakai lewat CLI) |
| `evaluate_images.py` | Evaluasi model dengan gambar baru per folder kelas |
| `best_cifar10_tuned.keras` | Model CNN hasil optimasi |
| `test_images/` | 50 gambar uji baru (5 per kelas) + `SOURCES.csv` (sumber & lisensi) |
| `Dockerfile` | Image untuk Hugging Face Spaces / Docker lokal |
| `tests/` | Unit test (pytest) |

## 1. Optimasi Model

Konfigurasi di bawah dibaca langsung dari file `best_cifar10_tuned.keras`
(`model.summary()` dan `model.optimizer.get_config()`).

Arsitektur `cnn_optimized` (input 32×32×3, 2.197.962 parameter trainable):

| Blok | Lapisan |
|---|---|
| Augmentasi (aktif hanya saat training) | `RandomFlip("horizontal")`, `RandomTranslation(0.125, 0.125, fill_mode="reflect")` |
| Blok 1 | 2 × [Conv2D 64 (3×3, same) → BatchNorm → ReLU] → MaxPool 2×2 → Dropout 0.1 |
| Blok 2 | 2 × [Conv2D 128 → BatchNorm → ReLU] → MaxPool 2×2 → Dropout 0.1 |
| Blok 3 | 2 × [Conv2D 256 → BatchNorm → ReLU] → MaxPool 2×2 → Dropout 0.1 |
| Klasifikasi | Flatten → Dense 256 → BatchNorm → ReLU → Dropout 0.3 → Dense 10 (softmax) |

Hyperparameter training final:

- Optimizer: SGD, momentum 0.9 (Nesterov), weight decay 1e-4
- Learning rate: warmup linear ke 1e-3 (3.828 step), lalu cosine decay (alpha 1e-3)
- Loss: sparse categorical cross-entropy
- Teknik regularisasi: dropout, batch normalization, weight decay, data augmentation
- Transfer learning tidak dipakai; model dilatih dari awal pada CIFAR-10

Akurasi pada test set CIFAR-10: 91% (hasil dari notebook training di Colab).

## 2. Menjalankan Aplikasi

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Buka http://localhost:8501, upload gambar JPG/PNG/WEBP. Aplikasi menampilkan
kelas prediksi, confidence, gambar 32×32 yang "dilihat" model, dan grafik
probabilitas tiap kelas. Jika confidence < 50%, muncul peringatan.

Prediksi lewat CLI:

```powershell
python model_utils.py test_images\cat\cat_03.jpg
```

Unit test: `pip install -r requirements-dev.txt` lalu `python -m pytest -q`.

## 3. Pengujian dengan Gambar Baru

### Metode

- 50 gambar baru dari Wikimedia Commons (lisensi CC / public domain), 5 per kelas.
  Sumber, penulis, dan lisensi tiap gambar ada di `test_images/SOURCES.csv`.
- Gambar sengaja bervariasi: ada yang objeknya besar di tengah, ada yang objeknya
  kecil di lanskap, latar ramai, atau foto hitam-putih.
- Preprocessing sama dengan training: RGB, resize 32×32, nilai piksel / 255.
- Perintah: `python evaluate_images.py --dir test_images --out hasil_uji.csv`

### Hasil

Akurasi total: 74% (37/50), lebih rendah dari akurasi test set CIFAR-10 (91%).

| Kelas | Benar | Akurasi |
|---|---|---|
| airplane | 4/5 | 80% |
| automobile | 5/5 | 100% |
| bird | 3/5 | 60% |
| cat | 3/5 | 60% |
| deer | 5/5 | 100% |
| dog | 2/5 | 40% |
| frog | 5/5 | 100% |
| horse | 3/5 | 60% |
| ship | 4/5 | 80% |
| truck | 3/5 | 60% |

Confusion matrix (baris = label asli, kolom = prediksi):

```
asli \ pred airpl autom  bird   cat  deer   dog  frog horse  ship truck
airplane        4     .     1     .     .     .     .     .     .     .
automobile      .     5     .     .     .     .     .     .     .     .
bird            .     .     3     .     .     .     1     1     .     .
cat             .     .     .     3     2     .     .     .     .     .
deer            .     .     .     .     5     .     .     .     .     .
dog             .     .     1     .     .     2     .     2     .     .
frog            .     .     .     .     .     .     5     .     .     .
horse           1     .     .     .     1     .     .     3     .     .
ship            1     .     .     .     .     .     .     .     4     .
truck           2     .     .     .     .     .     .     .     .     3
```

Prediksi yang salah:

| Gambar | Prediksi | Confidence | Kondisi gambar |
|---|---|---|---|
| airplane_05 | bird | 70.7% | Foto lama hitam-putih, latar pegunungan |
| bird_03 | frog | 73.4% | Burung kecil tertutup ranting |
| bird_05 | horse | 55.8% | Burung di antara ranting cokelat |
| cat_01 | deer | 33.2% | Kucing merunduk, latar dinding |
| cat_02 | deer | 93.2% | Kucing di rumput |
| dog_02 | horse | 75.9% | Anjing berkaki panjang, berdiri |
| dog_03 | bird | 78.9% | Anjing kecil di tengah rumput |
| dog_04 | horse | 82.5% | Anjing berbaring di bawah kursi |
| horse_04 | airplane | 99.7% | Kuda kecil, sebagian besar frame langit |
| horse_05 | deer | 59.0% | Kuda putih dengan latar gunung |
| ship_04 | airplane | 97.4% | Kapal kecil, langit dan gunung dominan |
| truck_02 | airplane | 57.3% | Truk kecil di lanskap |
| truck_05 | airplane | 81.4% | Truk kecil, langit dominan |

### Temuan

1. Ukuran objek dalam frame paling berpengaruh. Hampir semua kesalahan terjadi
   pada gambar dengan objek kecil. Setelah diperkecil ke 32×32, objek tersebut
   tinggal beberapa piksel, sehingga yang tersisa hanya latar.
2. Model memakai latar sebagai petunjuk. Lima gambar dengan langit dominan
   diprediksi airplane (kuda, kapal, dua truk), dan kucing di rumput
   diprediksi deer. Pola ini wajar karena di CIFAR-10 pesawat umumnya berlatar
   langit dan rusa berlatar rumput.
3. Hewan berkaki empat paling sering tertukar (cat, dog, deer, horse), sama
   seperti pola kesalahan umum pada CIFAR-10. Kelas dengan bentuk/warna khas
   (automobile, frog) dan foto close-up rusa mencapai 100%.
4. Model sering terlalu yakin saat salah. Rata-rata confidence prediksi benar
   89.9%, prediksi salah 73.7%, dan hanya 1 dari 13 kesalahan berada di bawah
   ambang 50%. Jadi peringatan confidence rendah di aplikasi membantu, tetapi
   tidak menjamin prediksi di atas 50% pasti benar. Pada confidence ≥ 90%,
   27 dari 30 prediksi benar.
5. Penurunan dari 91% ke 74% menunjukkan adanya domain shift: foto asli
   beresolusi tinggi dengan komposisi bebas berbeda dari gambar CIFAR-10 yang
   kecil dan objeknya terpusat.

Saran pengembangan: crop objek sebelum prediksi (atau pakai object detector),
transfer learning dari model pretrained beresolusi lebih tinggi (misalnya
ResNet/EfficientNet dari ImageNet), augmentasi yang lebih kuat (random crop/zoom,
color jitter), dan kalibrasi confidence (temperature scaling).

## 4. Deploy

### Streamlit Community Cloud

1. Push folder ini ke repository GitHub (file model 16.9 MB, di bawah batas 100 MB GitHub).
2. Di share.streamlit.io pilih repo, branch, dan main file `app.py`.
3. Di "Advanced settings", pilih Python 3.13 (atau 3.12). TensorFlow 2.21 hanya
   menyediakan wheel untuk Python 3.10–3.13, jadi Python 3.14 akan gagal install.

Kendala yang ditemui saat deploy: percobaan pertama gagal dengan
`Error installing requirements`. Log build menunjukkan environment memakai
Python 3.14.0, sehingga pip tidak menemukan wheel TensorFlow:

```
ERROR: Could not find a version that satisfies the requirement tensorflow-cpu==2.21.0
ERROR: No matching distribution found for tensorflow-cpu==2.21.0
```

Solusinya mengubah versi Python app menjadi 3.13 lalu deploy ulang. Versi Python
tidak bisa diatur lewat `runtime.txt` di Community Cloud, hanya lewat
"Advanced settings" di UI.

### Hugging Face Spaces (Docker)

SDK Streamlit bawaan HF Spaces sudah deprecated, jadi dipakai Docker SDK.
Metadata di bagian atas README ini (`sdk: docker`, `app_port: 7860`) dibaca otomatis oleh HF.

1. Buat Space baru dengan SDK "Docker" (template Blank).
2. Upload/push `README.md`, `Dockerfile`, `requirements.txt`, `.streamlit/`,
   `app.py`, `model_utils.py`, dan `best_cifar10_tuned.keras`.
3. Space akan build otomatis lalu tersedia di `https://<user>-<space>.hf.space`.

`Dockerfile` mematikan XSRF protection karena upload file Streamlit gagal di dalam
iframe HF Spaces jika opsi itu aktif. Aplikasi ini tidak punya login atau data
pengguna, jadi risikonya kecil, tetapi siapa pun yang punya URL bisa memakainya.

### Docker lokal

```powershell
docker build -t cifar10-app .
docker run --rm -p 7860:7860 cifar10-app
```

Buka http://localhost:7860.
