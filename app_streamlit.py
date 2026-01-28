import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="Prediksi Nilai TKA",
    page_icon="📊"
)


model = joblib.load("model.joblib")


st.title("📊 Prediksi Nilai TKA Siswa SMK N 1 PURBALINGGA")
st.markdown(
    "Aplikasi **Machine Learning Regresi Linear** untuk memprediksi **Nilai TKA Siswa** "
    "berdasarkan jam belajar, kehadiran, dan bimbel."
)

st.divider()

# ===========
# INPUT DATA
# ===========
st.subheader("📝 Input Data Siswa")

jam_belajar = st.slider(
    "Jam Belajar per Hari",
    min_value=0.0,
    max_value=10.0,
    value=4.0,
    step=0.5
)

persen_kehadiran = st.slider(
    "Persentase Kehadiran (%)",
    min_value=0,
    max_value=100,
    value=80
)

bimbel = st.selectbox(
    "Mengikuti Bimbel?",
    ["ya", "tidak"]
)

# =========
# PREDIKSI
# =========
if st.button("Prediksi Nilai TKA", type="primary"):
    data_baru = pd.DataFrame(
        [[jam_belajar, persen_kehadiran, bimbel]],
        columns=["jam_belajar_per_hari", "persen_kehadiran", "bimbel"]
    )

    prediksi = model.predict(data_baru)[0]
    prediksi = max(0, min(100, prediksi))  # Batasi nilai 0–100

    st.success(f"📈 Prediksi Nilai TKA: **{prediksi:.0f}**")
    st.progress(prediksi / 100)

st.divider()

# =====================
# VISUALISASI KORELASI
# =====================
st.subheader("📊 Korelasi Data (Training)")

st.markdown(
    "Heatmap berikut menunjukkan hubungan antara jam belajar, kehadiran, dan nilai TKA."
)


contoh_df = pd.DataFrame({
    "jam_belajar_per_hari": [2, 4, 6, 8, 5],
    "persen_kehadiran": [60, 75, 85, 95, 80],
    "nilai_tka": [55, 65, 78, 90, 72]
})

num_features = ["jam_belajar_per_hari", "persen_kehadiran", "nilai_tka"]

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(contoh_df[num_features].corr(), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")

st.pyplot(fig)

st.divider()
st.caption("Dibuat oleh **Aden Bagus Susilo** • Kelas 11 Rekayasa Perangkat Lunak 1")
