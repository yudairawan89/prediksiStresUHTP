import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from sklearn.preprocessing import StandardScaler
import datetime
from io import BytesIO

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
th, td {
    text-align: center;
    padding: 8px;
    border: 1px solid #ddd;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 10px;
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

# === LOAD DAN BERSIHKAN DATA ===
df = load_data()
df.columns = df.columns.str.strip()

df = df.rename(columns={
    'Suhu (°C)': 'Temperature',
    'SpO2 (%)': 'SpO2',
    'HeartRate (BPM)': 'HeartRate',
    'SYS': 'SYS',
    'DIA': 'DIA'
})

expected_columns = ['Temperature', 'SpO2', 'HeartRate', 'SYS', 'DIA']
df_clean = df[expected_columns].copy()

for col in expected_columns:
    df_clean[col] = df_clean[col].astype(str).str.replace(',', '.').astype(float).fillna(0)

X_scaled = scaler.transform(pd.DataFrame(df_clean.values, columns=scaler.feature_names_in_))
predictions = model.predict(X_scaled)
label_map = {0: 'Anxious', 1: 'Calm', 2: 'Relaxed', 3: 'Tense'}
df['Predicted Stress'] = [label_map.get(p, "Unknown") for p in predictions]

# === TAMPILKAN HASIL TERAKHIR DALAM TABEL RAPI ===
st.markdown("### 🔍 Hasil Prediksi Terakhir")
latest = df.iloc[-1]

data_tabel = pd.DataFrame({
    "Variabel": ['Temperature (°C)', 'SpO2 (%)', 'HeartRate (BPM)', 'SYS', 'DIA'],
    "Value": [
        latest['Temperature'],
        latest['SpO2'],
        latest['HeartRate'],
        latest['SYS'],
        latest['DIA']
    ]
})

st.markdown(data_tabel.to_html(index=False, escape=False), unsafe_allow_html=True)

st.markdown(f"""
<p style='font-size: 18px; background-color:#f0f0f0;
padding:10px; border-radius:5px; text-align:center;'>
<b>Predicted Stress Level:</b> <span style='font-size: 22px;'>{latest['Predicted Stress']}</span>
</p>
""", unsafe_allow_html=True)

# === TAMPILKAN DATA LENGKAP ===
st.markdown("<div class='section-title'>📊 Data Lengkap dan Prediksi</div>", unsafe_allow_html=True)
st.dataframe(df, use_container_width=True)

# === DOWNLOAD BUTTON ===
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

# === PENGUJIAN MANUAL ===
st.markdown("<div class='section-title'>🧪 Pengujian Menggunakan Data Manual</div>", unsafe_allow_html=True)

if "manual_input" not in st.session_state:
    st.session_state.manual_input = {
        "Temperature": 36.5,
        "SpO2": 98.0,
        "HeartRate": 90,
        "SYS": 120,
        "DIA": 80
    }
if "manual_result" not in st.session_state:
    st.session_state.manual_result = None

col1, col2, col3 = st.columns(3)
with col1:
    suhu = st.number_input("Temperature (°C)", value=st.session_state.manual_input["Temperature"])
    spo2 = st.number_input("SpO2 (%)", value=st.session_state.manual_input["SpO2"])
with col2:
    hr = st.number_input("HeartRate (BPM)", value=st.session_state.manual_input["HeartRate"])
    sys = st.number_input("SYS", value=st.session_state.manual_input["SYS"])
with col3:
    dia = st.number_input("DIA", value=st.session_state.manual_input["DIA"])

btn1, btn2 = st.columns([1, 1])
with btn1:
    if st.button("🔍 Prediksi Manual"):
        input_df = pd.DataFrame([{
            "Temperature": suhu,
            "SpO2": spo2,
            "HeartRate": hr,
            "SYS": sys,
            "DIA": dia
        }])
        input_scaled = scaler.transform(pd.DataFrame(input_df.values, columns=scaler.feature_names_in_))
        pred = model.predict(input_scaled)[0]
        label = label_map.get(pred, "Unknown")
        st.session_state.manual_result = label
        st.session_state.manual_input = {
            "Temperature": suhu,
            "SpO2": spo2,
            "HeartRate": hr,
            "SYS": sys,
            "DIA": dia
        }

with btn2:
    if st.button("♻️ Reset Manual"):
        st.session_state.manual_input = {
            "Temperature": 36.5,
            "SpO2": 98.0,
            "HeartRate": 90,
            "SYS": 120,
            "DIA": 80
        }
        st.session_state.manual_result = None
        st.experimental_rerun()

if st.session_state.manual_result:
    hasil = st.session_state.manual_result
    st.markdown(f"""
        <p style='font-size: 18px; background-color:#d9f2d9;
        padding:10px; border-radius:5px; text-align:center;'>
        Hasil Prediksi Manual: <b>{hasil}</b>
        </p>
    """, unsafe_allow_html=True)
