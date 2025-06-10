import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from sklearn.preprocessing import StandardScaler
import datetime

# === PAGE CONFIG ===
st.set_page_config(page_title="StressSense - Realtime Stress Detection", layout="wide")

# === LOAD MODEL & SCALER ===
@st.cache_resource
def load_model_scaler():
    model, scaler, encoder = joblib.load("stacking_model_stres.joblib")
    return model, scaler, encoder

model, scaler, encoder = load_model_scaler()

# === STYLING ===
st.markdown("""
<style>
h1 { color: #2c3e50; }
.section-title {
    background-color: #3498db; padding: 10px; border-radius: 5px;
    color: white; font-size: 20px; font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# === LOAD REALTIME DATA ===
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1ftXpDVdOWA3nDothJGFxCBfZ-L2Ccr4SMBVyd4_C_v8/export?format=csv"
    return pd.read_csv(url)

st.title("🧠 StressSense: Real-Time Student Stress Detection")

st.markdown("<div class='section-title'>Live Monitoring (Auto-refresh setiap 7 detik)</div>", unsafe_allow_html=True)
st_autorefresh(interval=7000, key="refresh")

df = load_data()

# === RENAME SESUAI KOLON ===
df = df.rename(columns={
    'Suhu (°C)': 'Temperature',
    'SpO2 (%)': 'SpO2',
    'HeartRate (BPM)': 'HeartRate',
    'SYS': 'SYS',
    'DIA': 'DIA'
})

features = ['Temperature', 'SpO2', 'HeartRate', 'SYS', 'DIA']
df_clean = df[features].copy()

# Konversi tipe data dan tangani missing value
for col in features:
    df_clean[col] = df_clean[col].astype(str).str.replace(',', '.').astype(float).fillna(0)

# Prediksi
X_scaled = scaler.transform(df_clean)
predictions = model.predict(X_scaled)
label_map = {0: 'Anxious', 1: 'Calm', 2: 'Relaxed', 3: 'Tense'}
df['Predicted Stress'] = [label_map.get(p, "Unknown") for p in predictions]

# === TAMPILKAN HASIL TERAKHIR ===
st.markdown("### 🔍 Hasil Prediksi Terakhir")
latest = df.iloc[-1]
col1, col2 = st.columns(2)
with col1:
    st.write("**Waktu:**", latest['Timestamp'] if 'Timestamp' in latest else datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    st.write("**Temperature:**", latest['Temperature'])
    st.write("**SpO2:**", latest['SpO2'])
    st.write("**HeartRate:**", latest['HeartRate'])
    st.write("**SYS:**", latest['SYS'])
    st.write("**DIA:**", latest['DIA'])

with col2:
    st.markdown(f"<p style='font-size: 24px; background-color:#f0f0f0; padding:10px; border-radius:5px;'>Predicted Stress Level: <b>{latest['Predicted Stress']}</b></p>", unsafe_allow_html=True)

# === TAMPILKAN SELURUH DATA ===
st.markdown("<div class='section-title'>📊 Data Lengkap dan Prediksi</div>", unsafe_allow_html=True)
st.dataframe(df, use_container_width=True)

# === UNDUH DATA ===
from io import BytesIO
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

st.download_button(
    label="📥 Unduh Hasil Prediksi",
    data=to_excel(df),
    file_name="prediksi_stres.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
