# streamlit_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.preprocessing import StandardScaler

# === CONFIG === #
st.set_page_config(page_title="Stress Detection Dashboard", layout="wide")
st.title("📊 Real-Time Stress Detection Dashboard")

# === AUTHENTIKASI GOOGLE SHEETS === #
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    "credentials.json", scope
)
gc = gspread.authorize(credentials)

# === BACA DATA DARI GOOGLE SHEET === #
@st.cache_data(ttl=60)
def load_data():
    sheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1ftXpDVdOWA3nDothJGFxCBfZ-L2Ccr4SMBVyd4_C_v8/edit#gid=0")
    worksheet = sheet.sheet1
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    return df

# === PREPROCESSING DATA TERBARU === #
def preprocess_data(df):
    df = df.drop(columns=['Timestamp'], errors='ignore')
    df.columns = ['Suhu', 'SpO2', 'HeartRate', 'SYS', 'DIA']
    df = df.replace(0, np.nan).dropna()  # buang nilai 0
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return scaled, df

# === LOAD MODEL TERBAIK === #
model = joblib.load("model_stacking_LR_XGB_OPTUNA.pkl")
label_map = {0: 'Anxious', 1: 'Relaxed', 2: 'Tense', 3: 'Calm'}

# === LOAD DATA === #
data = load_data()

if data.empty:
    st.warning("Tidak ada data tersedia.")
else:
    X_scaled, df_original = preprocess_data(data)
    predictions = model.predict(X_scaled)
    df_original['Predicted Condition'] = [label_map[p] for p in predictions]

    # === TAMPILKAN DATA === #
    st.subheader("📥 Input Data (Latest)")
    st.dataframe(df_original.tail(10), use_container_width=True)

    # === VISUALISASI === #
    st.subheader("📈 Deteksi Kondisi Tingkat Stres")
    condition_count = df_original['Predicted Condition'].value_counts()
    st.bar_chart(condition_count)

    # === AUTO REFRESH === #
    st.experimental_rerun()
