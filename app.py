# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# === CONFIGURASI HALAMAN === #
st.set_page_config(page_title="Stress Detection - IoT Dashboard", layout="wide")

# === STYLE === #
st.markdown("""
    <style>
    .main {background-color: #F9F9F9;}
    .section-title {
        background-color: #006699;
        color: white;
        padding: 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# === LOAD MODEL === #
@st.cache_resource
def load_model():
    return joblib.load("model_stacking_dummy_pipeline.joblib")

model = load_model()
label_map = {0: 'Cemas', 1: 'Rileks', 2: 'Tegang', 3: 'Tenang'}
color_map = {
    "Cemas": "red",
    "Rileks": "green",
    "Tegang": "yellow",
    "Tenang": "blue"
}

# === DATA GOOGLE SHEET === #
@st.cache_data(ttl=60)
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1ftXpDVdOWA3nDothJGFxCBfZ-L2Ccr4SMBVyd4_C_v8/export?format=csv"
    return pd.read_csv(url)

# === JUDUL === #
st.title("📡 Real-Time Stress Detection Dashboard")
st.markdown("""
<div class='section-title'>Prediksi Tingkat Stres Mahasiswa Berdasarkan Sinyal Fisiologis IoT</div>
""", unsafe_allow_html=True)

# === REFRESH OTOMATIS === #
st_autorefresh(interval=60000, key="auto_refresh")

# === PROSES DATA === #
df = load_data()

if df.empty:
    st.warning("Belum ada data tersedia dari perangkat IoT.")
else:
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    df.columns = ['Suhu', 'SpO2', 'HeartRate', 'SYS', 'DIA']
    df = df.replace(0, np.nan).dropna()

    predictions = model.predict(df)
    labels = [label_map[p] for p in predictions]
    df['Predicted Condition'] = labels

    # === VISUALISASI TABEL === #
    st.subheader("📋 Data Input dan Prediksi")
    st.dataframe(df.tail(10), use_container_width=True)

    # === BAR CHART KONDISI === #
    st.subheader("📊 Distribusi Tingkat Stres")
    count_chart = df['Predicted Condition'].value_counts()
    st.bar_chart(count_chart)

    # === STATUS TERAKHIR === #
    latest = df.iloc[-1]
    kondisi = latest['Predicted Condition']
    warna = color_map[kondisi]
    st.markdown(
        f"""
        <div style='background-color:{warna}; padding:15px; border-radius:10px; color:white;'>
        <h4>🧠 Kondisi Terakhir Terdeteksi: <b>{kondisi}</b></h4>
        </div>
        """, unsafe_allow_html=True
    )

    # === DOWNLOAD HASIL === #
    from io import BytesIO
    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
        writer.close()
        return output.getvalue()

    st.download_button(
        label="⬇️ Download Hasil Prediksi",
        data=to_excel(df),
        file_name="hasil_prediksi_stres.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
