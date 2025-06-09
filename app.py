import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# ================================
# LOAD MODEL & PREPROCESSOR
# ================================
model = joblib.load("model_stacking_dbd.pkl")
scaler = joblib.load("scaler_dbd.pkl")
le = joblib.load("label_encoder_dbd.pkl")

# ================================
# KOORDINAT DEFAULT PER KECAMATAN
# ================================
kecamatan_coords = {
    "Sukajadi": [-0.5176592, 101.4367539],
    "Senapelan": [-0.5361938, 101.4367539],
    "Pekanbaru Kota": [-0.5070677, 101.4477793],
    "Rumbai Pesisir": [-0.6079579, 101.5020752],
    "Rumbai": [-0.6453205, 101.4112049],
    "Lima Puluh": [-0.2490762, 100.6120232],
    "Sail": [-0.5176177, 101.4594696],
    "Bukit Raya": [-0.4689961, 101.4679893],
    "Marpoyan Damai": [-0.4736702, 101.4395931],
    "Tenayan Raya": [-0.4966231, 101.5475409],
    "Tampan": [-0.4691089, 101.3998518],
    "Payung Sekaki": [-0.5246769, 101.3998518]
}

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="Prediksi Risiko DBD", layout="wide")
st.markdown("""
    <h1 style='color:#0056b3;'>📊 Dashboard Prediksi Risiko DBD Berbasis Machine Learning</h1>
    <p style='font-size:16px'>Alat bantu ini dirancang untuk mendeteksi tingkat risiko DBD berdasarkan data lingkungan, cuaca, dan sosial per wilayah. Silakan unggah data dalam format <b>.csv</b>.</p>
""", unsafe_allow_html=True)

# ================================
# UPLOAD DATA
# ================================
uploaded_file = st.file_uploader("Unggah file CSV", type=["CSV"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Data yang Diupload")
    st.dataframe(df.head())

    fitur = [
        'jumlah_kasus_dbd', 'curah_hujan', 'jumlah_tps_liar',
        'suhu_rata_rata', 'jumlah_fogging', 'jumlah_genangan_air',
        'kelembaban', 'pengangguran', 'tingkat_pendidikan'
    ]

    try:
        X = df[fitur]
        X_scaled = scaler.transform(X)
        prediksi = model.predict(X_scaled)
        prediksi_label = le.inverse_transform(prediksi)
        df['Prediksi Risiko DBD'] = prediksi_label

        df_prediksi_only = df[['kecamatan', 'Prediksi Risiko DBD']].copy()

        def rekomendasi(label):
            if label == 'Tinggi':
                return [
                    "Fogging Massal Terjadwal: Lakukan pengasapan minimal dua kali seminggu di seluruh area kecamatan dengan pengawasan oleh Dinas Kesehatan.",
                    "Peningkatan Pemberantasan Sarang Nyamuk (PSN): Dorong masyarakat untuk melakukan 3M Plus secara kolektif.",
                    "Posko Tanggap DBD: Bentuk tim siaga RT/RW dengan pelaporan kasus gejala demam tinggi dalam 24 jam.",
                    "Edukasi Intensif: Laksanakan penyuluhan door-to-door dan media sosial dengan pesan kunci seputar gejala, pencegahan, dan penanganan dini.",
                    "Pemeriksaan Jentik Berkala: Lakukan oleh kader Jumantik dan petugas Puskesmas minimal dua kali per bulan.",
                    "Skrining Kesehatan Sekolah: Wajibkan inspeksi jentik dan distribusi brosur edukasi DBD di sekolah-sekolah.",
                    "Koordinasi Lintas Sektor: Libatkan Lurah, Babinsa, dan tokoh masyarakat untuk gerakan pembersihan masif tiap akhir pekan."
                ]
            elif label == 'Sedang':
                return [
                    "Fogging Selektif: Lakukan pengasapan di lokasi dengan kasus baru atau potensi genangan.",
                    "Penguatan Edukasi RT/RW: Distribusi leaflet dan penyuluhan tentang pencegahan mandiri dan deteksi dini.",
                    "Pemantauan TPS Liar: Lakukan inspeksi lokasi pembuangan sampah sembarangan dan rencanakan penutupan/pemindahan.",
                    "Monitoring Genangan Air: Evaluasi sistem drainase dan upaya membersihkan saluran tersumbat.",
                    "Surveilans Aktif: Optimalkan pencatatan dan pelaporan dari Puskesmas dan rumah sakit.",
                    "Kolaborasi dengan Sekolah: Promosikan lomba kebersihan lingkungan dan pemantauan jentik di kelas."
                ]
            else:
                return [
                    "Monitoring Berkala: Pertahankan kegiatan Jumantik mingguan dan pelaporan digital bila tersedia.",
                    "Kampanye Preventif Ringan: Gunakan media komunitas dan masjid untuk mengingatkan pentingnya pencegahan DBD.",
                    "Survei Kesiapsiagaan Komunitas: Evaluasi kesiapan warga dan kader jika terjadi lonjakan kasus.",
                    "Evaluasi Infrastruktur: Pastikan tidak ada potensi TPS liar baru atau aliran air tersumbat yang bisa menjadi tempat nyamuk berkembang.",
                    "Penguatan Komunikasi Risiko: Sediakan papan informasi risiko DBD di kantor kelurahan dan puskesmas."
                ]

        df['Rekomendasi'] = df['Prediksi Risiko DBD'].apply(rekomendasi)
        df['latitude'] = df['kecamatan'].map(lambda x: kecamatan_coords.get(x, [0, 0])[0])
        df['longitude'] = df['kecamatan'].map(lambda x: kecamatan_coords.get(x, [0, 0])[1])

        output = df[['kecamatan', 'latitude', 'longitude', 'Prediksi Risiko DBD', 'Rekomendasi']].copy()
        output.insert(0, 'No', range(1, len(output) + 1))

        st.subheader("📌 Tabel Prediksi Risiko DBD")
        st.dataframe(df_prediksi_only)

        with st.expander("📋 Rekomendasi Tindakan Berdasarkan Tingkat Risiko DBD Per Kecamatan"):
            for _, row in output.iterrows():
                warna = {
                    'Tinggi': '#d9534f',
                    'Sedang': '#f0ad4e',
                    'Rendah': '#5cb85c'
                }.get(row['Prediksi Risiko DBD'], 'gray')

                st.markdown(f"""
                <details>
                <summary><strong>{row['kecamatan']} — Risiko: <span style='color:{warna}; font-weight:bold'>{row['Prediksi Risiko DBD']}</span></strong></summary>
                <div style='background-color:#f9f9f9; padding: 0.7rem 1rem; border-left: 5px solid {warna}; border-radius: 6px; margin-top: 0.5rem'>
                    <b>🧾 Detail Data:</b>
                    <ul>
                        <li>Jumlah Kasus DBD: {df.loc[_,'jumlah_kasus_dbd']}</li>
                        <li>Curah Hujan: {df.loc[_,'curah_hujan']} mm</li>
                        <li>Suhu Rata-rata: {df.loc[_,'suhu_rata_rata']} °C</li>
                        <li>Genangan Air: {df.loc[_,'jumlah_genangan_air']}</li>
                        <li>Pengangguran: {df.loc[_,'pengangguran']} %</li>
                        <li>Pendidikan: {df.loc[_,'tingkat_pendidikan']} tahun rata-rata</li>
                    </ul>
                    <b>📌 Rekomendasi Tindakan:</b>
                    <ol>
                        {''.join([f'<li>{s}</li>' for s in row['Rekomendasi']])}
                    </ol>
                </div>
                </details>
                """, unsafe_allow_html=True)

        st.markdown("### 🗺️ Visualisasi Peta Risiko DBD")
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11)

        color_map = {
            'Rendah': 'green',
            'Sedang': 'orange',
            'Tinggi': 'red'
        }

        for _, row in output.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=f"{row['kecamatan']}\nRisiko: {row['Prediksi Risiko DBD']}\n{'; '.join(row['Rekomendasi'])}",
                color=color_map.get(row['Prediksi Risiko DBD'], 'blue'),
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.DivIcon(html=f"""
                    <div style='font-size: 10pt; color: black'><b>{row['kecamatan']}</b></div>
                """),
            ).add_to(m)

        st_data = st_folium(m, width=800, height=500)

        csv = output.drop(columns=['latitude', 'longitude']).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Prediksi (CSV)",
            data=csv,
            file_name="hasil_prediksi_risiko_dbd.csv",
            mime="text/csv",
        )

    except KeyError:
        st.error("Kolom pada file CSV tidak sesuai. Pastikan menyertakan kolom: kecamatan dan fitur model.")
        st.markdown("### Kolom yang diperlukan:")
        st.code(", ".join(fitur + ['kecamatan']))

else:
    st.info("Silakan unggah data untuk memulai prediksi.")
