import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from sklearn.preprocessing import StandardScaler
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

st.title("🧠 UHTP Real-Time Student Stress Detection")
st.markdown("<div class='section-title'>Hasil Deteksi Tingkat Stres Secara Real Time</div>", unsafe_allow_html=True)
st_autorefresh(interval=4000, key="refresh")

# === BACA & BERSIHKAN DATA ===
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
label_map = {0: 'Anxious', 1: 'Calm', 2: 'Relaxed', 3: 'Tense'}
label_translate = {
    'Anxious': 'Cemas',
    'Calm': 'Tenang',
    'Relaxed': 'Relaks',
    'Tense': 'Tegang'
}
predictions = model.predict(X_scaled)
df['Predicted Stress'] = [label_map.get(p, "Unknown") for p in predictions]

# === TAMPILKAN HASIL TERAKHIR ===
st.markdown("### 🔍 Hasil Prediksi Terakhir")
latest = df.iloc[-1]
data_rows = [
    ('Temperature (°C)', latest['Temperature']),
    ('SpO2 (%)', latest['SpO2']),
    ('HeartRate (BPM)', latest['HeartRate']),
    ('SYS', latest['SYS']),
    ('DIA', latest['DIA'])
]

table_html = """
<table style="width: 100%; background-color: #f9f9f9; border-collapse: collapse; margin-top: 10px; margin-bottom: 15px;">
    <thead>
        <tr style="background-color: #e0e0e0;">
            <th style="text-align: center; padding: 10px;">Variabel</th>
            <th style="text-align: center; padding: 10px;">Value</th>
        </tr>
    </thead>
    <tbody>
"""

for label, value in data_rows:
    table_html += f"""
        <tr>
            <td style="text-align: center; padding: 10px;">{label}</td>
            <td style="text-align: center; padding: 10px;">{value}</td>
        </tr>
    """

table_html += "</tbody></table>"
st.markdown(table_html, unsafe_allow_html=True)

st.markdown(f"""
<p style='font-size: 18px; background-color:#f0f0f0;
padding:10px; border-radius:5px; text-align:center;'>
<b>Predicted Stress Level:</b> <span style='font-size: 22px;'>{latest['Predicted Stress']} / {label_translate.get(latest['Predicted Stress'], "-")}</span>
</p>
""", unsafe_allow_html=True)

# === TAMPILKAN DATA LENGKAP ===
st.markdown("<div class='section-title'>📊 Deteksi Stres Data Kolektif</div>", unsafe_allow_html=True)
st.dataframe(df, use_container_width=True)

# === DOWNLOAD ===
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

st.download_button(
    label="📥 Unduh Hasil Deteksi Tingkat Stres",
    data=to_excel(df),
    file_name="prediksi_stres.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# === PENGUJIAN MANUAL ===
st.markdown("<div class='section-title'>🧪 Pengujian Menggunakan Data Manual</div>", unsafe_allow_html=True)

if "manual_input" not in st.session_state:
    st.session_state.manual_input = {
        "Temperature": 0.0,
        "SpO2": 0.0,
        "HeartRate": 0.0,
        "SYS": 0.0,
        "DIA": 0.0
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
            "Temperature": 0.0,
            "SpO2": 0.0,
            "HeartRate": 0.0,
            "SYS": 0.0,
            "DIA": 0.0
        }
        st.session_state.manual_result = None
        st.rerun()

if st.session_state.manual_result:
    hasil = st.session_state.manual_result
    st.markdown(f"""
        <p style='font-size: 18px; background-color:#d9f2d9;
        padding:10px; border-radius:5px; text-align:center;'>
        Hasil Prediksi Manual: <b>{hasil} / {label_translate.get(hasil, "-")}</b>
        </p>
    """, unsafe_allow_html=True)
