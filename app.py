import altair as alt
import streamlit as st
from PIL import Image, UnidentifiedImageError

from model_utils import (
    CLASS_NAMES,
    IMG_SIZE,
    build_prob_df,
    load_model,
    predict,
    resolve_model_path,
    to_model_view,
)

LOW_CONFIDENCE = 0.50
PREVIEW_SIZE = 256

st.set_page_config(page_title="Klasifikasi Gambar", page_icon="🖼️", layout="centered")


@st.cache_resource(show_spinner="Memuat model CNN...")
def get_model(path: str):
    # Cache per path: model hanya dimuat sekali selama server berjalan.
    return load_model(path)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Tentang Model")
    st.markdown(
        "- Arsitektur: CNN (Conv2D + BatchNorm + Dropout)\n"
        "- Dataset: CIFAR-10\n"
        f"- Ukuran input: {IMG_SIZE[0]}×{IMG_SIZE[1]} piksel RGB\n"
        "- Akurasi test: 91%"
    )
    st.subheader("Kelas yang dikenali")
    st.markdown("\n".join(f"{i}. {name}" for i, name in enumerate(CLASS_NAMES)))
    st.subheader("Tips")
    st.markdown(
        "- Pakai gambar dengan satu objek utama.\n"
        "- Usahakan objek berada di tengah dan memenuhi frame.\n"
        "- Objek di luar 10 kelas di atas tetap akan dipaksa masuk salah satu kelas."
    )

# ---------- Halaman utama ----------
st.title("Klasifikasi Gambar CIFAR-10")
st.write(
    "Upload sebuah gambar, lalu model CNN akan memprediksi termasuk kelas apa "
    "dari 10 kelas CIFAR-10 beserta probabilitas tiap kelas."
)

model_path = resolve_model_path()
try:
    model = get_model(str(model_path))
except FileNotFoundError:
    st.error(
        f"File model tidak ditemukan di `{model_path}`. "
        "Taruh file `best_cifar10_tuned.keras` di folder aplikasi, "
        "atau set environment variable `MODEL_PATH` ke lokasi file model."
    )
    st.stop()

uploaded = st.file_uploader(
    "Pilih gambar", type=["jpg", "jpeg", "png", "webp"], help="Format: JPG, PNG, atau WEBP"
)

if uploaded is None:
    st.info("Belum ada gambar. Silakan upload gambar untuk memulai prediksi.")
    st.stop()

try:
    image = Image.open(uploaded)
    image.load()
except (UnidentifiedImageError, OSError):
    st.error("File yang di-upload bukan gambar yang valid. Coba file lain.")
    st.stop()

with st.spinner("Memprediksi..."):
    results = predict(model, image)

top_class, top_prob = results[0]

col_img, col_view = st.columns(2)
with col_img:
    st.image(image, caption="Gambar asli", width="stretch")
with col_view:
    # Perbesar 32x32 dengan NEAREST supaya pikselnya terlihat jelas.
    model_view = to_model_view(image).resize(
        (PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.NEAREST
    )
    st.image(model_view, caption="Yang dilihat model (32×32)", width="stretch")

st.metric("Prediksi", top_class, help="Kelas dengan probabilitas tertinggi")
st.metric("Confidence", f"{top_prob:.2%}")

if top_prob < LOW_CONFIDENCE:
    st.warning(
        f"Model kurang yakin (confidence di bawah {LOW_CONFIDENCE:.0%}). "
        "Kemungkinan objek di luar 10 kelas CIFAR-10, ada beberapa objek, "
        "atau detail hilang saat gambar diperkecil ke 32×32."
    )

st.subheader("Probabilitas Tiap Kelas")
prob_df = build_prob_df(results)
chart = (
    alt.Chart(prob_df)
    .mark_bar()
    .encode(
        x=alt.X("probabilitas:Q", title="Probabilitas", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("kelas:N", title=None, sort="-x"),
        color=alt.condition(
            alt.datum.kelas == top_class, alt.value("#e4572e"), alt.value("#4c78a8")
        ),
        tooltip=["kelas", alt.Tooltip("probabilitas:Q", format=".2%")],
    )
    .properties(height=320)
)
st.altair_chart(chart, width="stretch")

with st.expander("Lihat tabel probabilitas"):
    st.dataframe(
        prob_df,
        hide_index=True,
        column_config={
            "kelas": "Kelas",
            "probabilitas": st.column_config.ProgressColumn(
                "Probabilitas", format="percent", min_value=0.0, max_value=1.0
            ),
        },
    )
