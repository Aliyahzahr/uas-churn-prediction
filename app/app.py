import os
import time
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

# =====================================================
# ChurnGuard - UI Revisi untuk Deployment Streamlit
# =====================================================

st.set_page_config(
    page_title="ChurnGuard | Prediksi Customer Churn",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CUSTOM STYLE ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%);
    }

    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1rem;
        line-height: 1.6;
    }

    .card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 22px;
        padding: 22px 24px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
        margin-bottom: 18px;
    }

    .card-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 12px;
    }

    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 16px;
        height: 100%;
    }

    .small-muted {
        color: #64748B;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .risk-high {
        background: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        border-radius: 18px;
        padding: 18px;
        font-weight: 700;
    }

    .risk-medium {
        background: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        border-radius: 18px;
        padding: 18px;
        font-weight: 700;
    }

    .risk-low {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        border-radius: 18px;
        padding: 18px;
        font-weight: 700;
    }

    .feature-chip {
        display: inline-block;
        background: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
        border-radius: 999px;
        padding: 6px 10px;
        margin: 3px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SESSION STATE ----------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ---------- FEATURE DEFINITIONS ----------
SCALE_COLS = [
    "age",
    "total_visits",
    "avg_session_time",
    "pages_per_session",
    "email_open_rate",
    "email_click_rate",
    "total_spent",
    "avg_order_value",
    "support_tickets",
    "delivery_delay_days",
    "satisfaction_score",
    "nps_score",
    "marketing_spend_per_user",
    "lifetime_value",
    "last_3_month_purchase_freq",
    "tenure_days",
    "days_since_last_purchase",
]

FEATURE_INFO = {
    "age": {
        "label": "Usia Pelanggan",
        "simple": "Umur pelanggan.",
        "why": "Usia bisa berkaitan dengan pola belanja dan preferensi layanan.",
        "example": "Contoh: 35 tahun",
    },
    "total_visits": {
        "label": "Total Kunjungan",
        "simple": "Berapa kali pelanggan mengunjungi aplikasi/website.",
        "why": "Semakin sering berkunjung, biasanya pelanggan lebih aktif.",
        "example": "Contoh: 12 kunjungan",
    },
    "avg_session_time": {
        "label": "Durasi Sesi Rata-rata",
        "simple": "Rata-rata lama pelanggan menggunakan aplikasi setiap kunjungan.",
        "why": "Durasi rendah bisa menandakan engagement yang kurang.",
        "example": "Contoh: 8 menit",
    },
    "pages_per_session": {
        "label": "Halaman per Sesi",
        "simple": "Rata-rata jumlah halaman yang dibuka dalam satu sesi.",
        "why": "Semakin banyak halaman dibuka, semakin tinggi interaksi pelanggan.",
        "example": "Contoh: 4 halaman/sesi",
    },
    "email_open_rate": {
        "label": "Email Open Rate",
        "simple": "Persentase email promosi/informasi yang dibuka pelanggan.",
        "why": "Jika rendah, pelanggan kurang tertarik dengan komunikasi brand.",
        "example": "0.30 artinya 30% email dibuka",
    },
    "email_click_rate": {
        "label": "Email Click Rate",
        "simple": "Persentase link dalam email yang diklik pelanggan.",
        "why": "Mengukur respons pelanggan terhadap kampanye email.",
        "example": "0.10 artinya 10% link diklik",
    },
    "total_spent": {
        "label": "Total Pengeluaran",
        "simple": "Total uang yang pernah dibelanjakan pelanggan.",
        "why": "Pelanggan dengan pengeluaran kecil bisa punya risiko churn berbeda dibanding pelanggan bernilai tinggi.",
        "example": "Contoh: 350",
    },
    "avg_order_value": {
        "label": "Rata-rata Nilai Pesanan",
        "simple": "Rata-rata nilai transaksi setiap kali pelanggan membeli.",
        "why": "Membantu melihat nilai ekonomi tiap transaksi pelanggan.",
        "example": "Contoh: 50 per pesanan",
    },
    "support_tickets": {
        "label": "Jumlah Tiket Keluhan",
        "simple": "Berapa kali pelanggan menghubungi support/komplain.",
        "why": "Keluhan tinggi sering menjadi sinyal ketidakpuasan.",
        "example": "Contoh: 2 tiket",
    },
    "delivery_delay_days": {
        "label": "Keterlambatan Pengiriman",
        "simple": "Rata-rata jumlah hari keterlambatan pengiriman.",
        "why": "Pengalaman pengiriman buruk dapat meningkatkan risiko churn.",
        "example": "Contoh: 2 hari",
    },
    "satisfaction_score": {
        "label": "Skor Kepuasan",
        "simple": "Nilai kepuasan pelanggan dari 1 sampai 5.",
        "why": "Ini salah satu indikator terkuat untuk prediksi churn.",
        "example": "1 = sangat tidak puas, 5 = sangat puas",
    },
    "nps_score": {
        "label": "NPS Score",
        "simple": "Kemungkinan pelanggan merekomendasikan layanan ke orang lain.",
        "why": "NPS rendah menandakan pelanggan kurang loyal.",
        "example": "0 = tidak mungkin, 10 = sangat mungkin",
    },
    "marketing_spend_per_user": {
        "label": "Biaya Marketing per User",
        "simple": "Biaya pemasaran yang dikeluarkan untuk pelanggan tersebut.",
        "why": "Membantu melihat hubungan biaya akuisisi/retensi dengan churn.",
        "example": "Contoh: 15",
    },
    "lifetime_value": {
        "label": "Lifetime Value",
        "simple": "Estimasi total nilai pelanggan selama memakai layanan.",
        "why": "Pelanggan bernilai tinggi perlu prioritas retensi lebih besar.",
        "example": "Contoh: 700",
    },
    "last_3_month_purchase_freq": {
        "label": "Frekuensi Beli 3 Bulan Terakhir",
        "simple": "Berapa kali pelanggan membeli dalam 3 bulan terakhir.",
        "why": "Frekuensi rendah bisa menjadi tanda pelanggan mulai tidak aktif.",
        "example": "Contoh: 3 kali",
    },
    "tenure_days": {
        "label": "Lama Menjadi Pelanggan",
        "simple": "Jumlah hari sejak pelanggan pertama kali terdaftar.",
        "why": "Pelanggan baru dan pelanggan lama bisa punya pola churn berbeda.",
        "example": "Contoh: 660 hari",
    },
    "days_since_last_purchase": {
        "label": "Hari Sejak Pembelian Terakhir",
        "simple": "Sudah berapa hari pelanggan tidak melakukan pembelian.",
        "why": "Semakin lama tidak membeli, biasanya risiko churn makin tinggi.",
        "example": "Contoh: 397 hari",
    },
}

PRESETS = {
    "Pelanggan aktif/loyal": {
        "age": 32,
        "tenure_days": 900,
        "days_since_last_purchase": 20,
        "last_3_month_purchase_freq": 12,
        "total_visits": 45,
        "avg_session_time": 14.5,
        "pages_per_session": 7.0,
        "email_open_rate": 0.65,
        "email_click_rate": 0.22,
        "support_tickets": 0,
        "total_spent": 1200.0,
        "avg_order_value": 85.0,
        "lifetime_value": 2400.0,
        "marketing_spend_per_user": 30.0,
        "delivery_delay_days": 0,
        "satisfaction_score": 5,
        "nps_score": 9,
    },
    "Pelanggan berisiko": {
        "age": 41,
        "tenure_days": 450,
        "days_since_last_purchase": 320,
        "last_3_month_purchase_freq": 0,
        "total_visits": 4,
        "avg_session_time": 3.5,
        "pages_per_session": 2.0,
        "email_open_rate": 0.08,
        "email_click_rate": 0.01,
        "support_tickets": 7,
        "total_spent": 90.0,
        "avg_order_value": 25.0,
        "lifetime_value": 160.0,
        "marketing_spend_per_user": 10.0,
        "delivery_delay_days": 8,
        "satisfaction_score": 2,
        "nps_score": 2,
    },
    "Pelanggan rata-rata": {
        "age": 35,
        "tenure_days": 660,
        "days_since_last_purchase": 120,
        "last_3_month_purchase_freq": 3,
        "total_visits": 12,
        "avg_session_time": 8.0,
        "pages_per_session": 4.0,
        "email_open_rate": 0.30,
        "email_click_rate": 0.10,
        "support_tickets": 2,
        "total_spent": 350.0,
        "avg_order_value": 50.0,
        "lifetime_value": 700.0,
        "marketing_spend_per_user": 15.0,
        "delivery_delay_days": 2,
        "satisfaction_score": 4,
        "nps_score": 8,
    },
}

# ---------- HELPER FUNCTIONS ----------
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    paths_to_try = [
        os.path.join(current_dir, "model"),
        os.path.join(current_dir, "notebook", "model"),
        os.path.join(current_dir, "..", "notebook", "model"),
        os.path.join(current_dir, "..", "model"),
        current_dir,
        ".",
    ]

    for path in paths_to_try:
        model_path = os.path.join(path, "best_model.pkl")
        scaler_path = os.path.join(path, "scaler.pkl")
        features_path = os.path.join(path, "selected_features.pkl")

        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            selected_features = joblib.load(features_path)
            return model, scaler, selected_features, path

    raise FileNotFoundError(
        "File model tidak ditemukan. Pastikan best_model.pkl, scaler.pkl, dan selected_features.pkl ada di folder model/."
    )


@st.cache_data
def load_raw_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    paths_to_try = [
        os.path.join(current_dir, "..", "data", "Sales - Marketing customer dataset.csv"),
        os.path.join(current_dir, "data", "Sales - Marketing customer dataset.csv"),
        os.path.join(current_dir, "notebook", "data", "Sales - Marketing customer dataset.csv"),
        "./data/Sales - Marketing customer dataset.csv",
        "data/Sales - Marketing customer dataset.csv",
    ]
    for p in paths_to_try:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError("Dataset CSV tidak ditemukan.")


def make_gauge(value):
    if value < 30:
        color = "#16A34A"
    elif value < 60:
        color = "#F59E0B"
    else:
        color = "#DC2626"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 40, "color": color}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "bgcolor": "#F8FAFC",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#DCFCE7"},
                    {"range": [30, 60], "color": "#FEF3C7"},
                    {"range": [60, 100], "color": "#FEE2E2"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=15, r=15, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def risk_label(probability):
    if probability < 30:
        return "Rendah", "risk-low", "Pelanggan cenderung masih aktif/loyal."
    if probability < 60:
        return "Sedang", "risk-medium", "Pelanggan perlu dipantau karena ada beberapa sinyal churn."
    return "Tinggi", "risk-high", "Pelanggan berpotensi tinggi berhenti/meninggalkan layanan."


def explain_input_signals(data):
    signals = []
    if data["satisfaction_score"] <= 2:
        signals.append("Skor kepuasan rendah, ini dapat menjadi tanda pelanggan tidak puas.")
    if data["support_tickets"] >= 5:
        signals.append("Jumlah tiket keluhan tinggi, perlu evaluasi kualitas layanan.")
    if data["days_since_last_purchase"] >= 180:
        signals.append("Pelanggan sudah lama tidak membeli, ada indikasi mulai tidak aktif.")
    if data["last_3_month_purchase_freq"] <= 1:
        signals.append("Frekuensi pembelian 3 bulan terakhir rendah.")
    if data["email_open_rate"] < 0.15:
        signals.append("Email open rate rendah, pelanggan kurang merespons komunikasi brand.")
    if data["nps_score"] <= 4:
        signals.append("NPS rendah, pelanggan kecil kemungkinan merekomendasikan layanan.")
    if not signals:
        signals.append("Tidak ada sinyal risiko ekstrem dari input utama.")
    return signals


def top_feature_chart(model, selected_features):
    if not hasattr(model, "feature_importances_"):
        return None
    df_imp = pd.DataFrame(
        {
            "Fitur": selected_features,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False).head(7)
    df_imp["Nama Mudah"] = df_imp["Fitur"].map(lambda x: FEATURE_INFO.get(x, {}).get("label", x))

    fig = go.Figure(
        go.Bar(
            x=df_imp["Importance"],
            y=df_imp["Nama Mudah"],
            orientation="h",
            marker_color="#6366F1",
            text=df_imp["Importance"].round(3),
            textposition="auto",
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis={"autorange": "reversed"},
        xaxis_title="Importance",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------- LOAD MODEL & DATA ----------
try:
    model, scaler, selected_features, loaded_from_path = load_assets()
    assets_loaded = True
except Exception as exc:
    assets_loaded = False
    load_error = str(exc)

try:
    df_raw = load_raw_data()
    data_loaded = True
except Exception as exc:
    data_loaded = False
    data_error = str(exc)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 🛡️ ChurnGuard")
    st.caption("Aplikasi prediksi customer churn berbasis model machine learning.")

    page = st.radio(
        "Menu",
        [
            "🏠 Dashboard Overview",
            "📊 Eksplorasi Data (EDA)",
            "🔮 Prediksi Churn",
            "📘 Panduan Fitur",
            "📈 Riwayat Analisis",
            "⚙️ Info Model",
        ],
    )

    st.divider()
    st.markdown("### 👩‍💻 Identitas")
    st.info("Aliyah Zahratu Rizqi\n\nNIM: A11.2023.15294")

    st.divider()
    if assets_loaded:
        st.success("Model berhasil dimuat")
    else:
        st.error("Model belum termuat")

# ---------- HEADER ----------
st.markdown('<div class="main-title">Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">UI revisi ini dibuat lebih sederhana: setiap input diberi penjelasan agar mudah dipahami, lalu model memprediksi apakah pelanggan berisiko churn atau tidak.</div>',
    unsafe_allow_html=True,
)

if not assets_loaded:
    st.error(load_error)
    st.stop()

# =====================================================
# PAGE: DASHBOARD OVERVIEW
# =====================================================
if page == "🏠 Dashboard Overview":
    st.markdown('<div class="card"><div class="card-title">🏠 Selamat Datang di ChurnGuard Dashboard!</div>', unsafe_allow_html=True)
    st.write(
        """
        Aplikasi **ChurnGuard** dirancang khusus untuk memprediksi risiko hilangnya pelanggan (*customer churn*)
        berdasarkan data aktivitas penggunaan, histori belanja, kepuasan pelanggan, dan pola interaksi email marketing.
        Proyek ini dikembangkan sebagai Tugas UAS Bengkel Koding Data Science, Universitas Dian Nuswantoro.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 4 KPI Metrics
    if data_loaded:
        total_cust = len(df_raw)
        churn_count = int(df_raw['churn'].sum())
        loyal_count = total_cust - churn_count
        churn_rate = (churn_count / total_cust) * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: center;">
                    <div style="font-size: 0.9rem; color: #64748B; font-weight: 600;">TOTAL PELANGGAN</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #1E1B4B; margin-top: 4px;">{total_cust:,}</div>
                    <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Dalam Database</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: center;">
                    <div style="font-size: 0.9rem; color: #64748B; font-weight: 600;">CHURN RATE</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #DC2626; margin-top: 4px;">{churn_rate:.2f}%</div>
                    <div style="font-size: 0.75rem; color: #EF4444; margin-top: 4px;">Pelanggan Berhenti</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: center;">
                    <div style="font-size: 0.9rem; color: #64748B; font-weight: 600;">PELANGGAN LOYAL</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #16A34A; margin-top: 4px;">{loyal_count:,}</div>
                    <div style="font-size: 0.75rem; color: #22C55E; margin-top: 4px;">Status Aktif (0)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: center;">
                    <div style="font-size: 0.9rem; color: #64748B; font-weight: 600;">PELANGGAN CHURN</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #B91C1C; margin-top: 4px;">{churn_count:,}</div>
                    <div style="font-size: 0.75rem; color: #EF4444; margin-top: 4px;">Status Berhenti (1)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.error(f"Gagal memuat statistik database pelanggan: {data_error}")

    # Brief Overview
    st.markdown('<div class="card"><div class="card-title">📖 Project Overview & Bisnis Konteks</div>', unsafe_allow_html=True)
    st.write(
        """
        **Mengapa memprediksi Churn sangat krusial?**
        - **Biaya Akuisisi Lebih Tinggi:** Mendapatkan pelanggan baru jauh lebih mahal (hingga 5x lipat) dibanding mempertahankan pelanggan yang sudah ada.
        - **Model Machine Learning yang Tepat:** Proyek ini menggunakan algoritma **Decision Tree Classifier (Tuned)** yang dilatih pada 15.000 data pelanggan dengan **17 fitur pilihan** (seperti Skor Kepuasan, Pengeluaran, Jumlah Komplain, dll.).
        - **Aksi Cepat:** Dashboard ini memprediksi secara real-time status risiko churn seorang pelanggan, dan memberikan saran taktis langsung kepada tim relasi pelanggan (CRM) dan pemasaran.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PAGE: EKSPLORASI DATA (EDA)
# =====================================================
elif page == "📊 Eksplorasi Data (EDA)":
    if not data_loaded:
        st.error(f"Dataset tidak dapat dimuat untuk EDA: {data_error}")
        st.stop()

    st.markdown('<div class="card"><div class="card-title">📊 Analisis Eksploratif Data (EDA)</div>', unsafe_allow_html=True)
    st.write(
        "Gunakan tab-tab di bawah ini untuk melihat hasil eksplorasi data yang sesuai dengan notebook pengerjaan."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📄 Tampilan Data & Deskripsi",
            "❓ Missing Value & Duplikasi",
            "🎯 Distribusi Churn (Target)",
            "📦 Analisis Outlier & Anomali",
            "🔥 Heatmap Korelasi",
        ]
    )

    with tab1:
        st.markdown("#### 5 Baris Pertama Dataset (`df.head()`) ")
        st.dataframe(df_raw.head(), use_container_width=True)

        st.markdown("#### Statistik Deskriptif Dataset (`df.describe()`)")
        st.dataframe(df_raw.describe(include='all'), use_container_width=True)

    with tab2:
        col_m1, col_m2 = st.columns([1, 1.2])
        with col_m1:
            st.markdown("#### Tabel Kolom dengan Missing Value")
            missing_count = df_raw.isnull().sum()
            missing_pct = (missing_count / len(df_raw) * 100).round(2)
            missing_df = pd.DataFrame({
                'Jumlah Missing': missing_count,
                'Persentase (%)': missing_pct
            }).sort_values('Persentase (%)', ascending=False)
            missing_filtered = missing_df[missing_df['Jumlah Missing'] > 0]
            st.dataframe(missing_filtered, use_container_width=True)
            
            st.markdown("#### Cek Duplikasi Baris")
            dup_count = df_raw.duplicated().sum()
            st.info(f"Jumlah baris duplikat di dataset: **{dup_count}**")

        with col_m2:
            st.markdown("#### Visualisasi Persentase Missing Value per Kolom")
            if len(missing_filtered) > 0:
                fig_missing = go.Figure(
                    go.Bar(
                        x=missing_filtered.index,
                        y=missing_filtered['Persentase (%)'],
                        text=missing_filtered['Persentase (%)'].map(lambda x: f"{x}%"),
                        textposition="outside",
                        marker_color=["#DC2626" if x > 20 else "#F59E0B" if x > 5 else "#3B82F6" for x in missing_filtered['Persentase (%)']],
                    )
                )
                fig_missing.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                    yaxis_title="Persentase Missing (%)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("Tidak ada missing value terdeteksi pada dataset.")

    with tab3:
        churn_counts = df_raw['churn'].value_counts()
        churn_pct = df_raw['churn'].value_counts(normalize=True) * 100
        dist_df = pd.DataFrame({
            'Jumlah': churn_counts,
            'Persentase (%)': churn_pct.round(2)
        })
        
        col_c1, col_c2 = st.columns([1, 1.5])
        with col_c1:
            st.markdown("#### Tabel Distribusi Target Churn")
            st.dataframe(dist_df, use_container_width=True)
            st.warning("⚠️ **Catatan Imbalanced Data:** Rasio Tidak Churn : Churn adalah sekitar **5.5 : 1**. Oleh karena itu, fokus evaluasi model dialihkan ke metrik F1-Score dan Recall kelas Churn (1)!")

        with col_c2:
            st.markdown("#### Visualisasi Distribusi Target")
            fig_target = go.Figure()
            fig_target.add_trace(
                go.Bar(
                    x=['Tidak Churn (0)', 'Churn (1)'],
                    y=churn_counts.values,
                    text=[f"{val:,} ({pct:.1f}%)" for val, pct in zip(churn_counts.values, churn_pct.values)],
                    textposition="auto",
                    marker_color=['#3B82F6', '#EF4444'],
                )
            )
            fig_target.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis_title="Jumlah Pelanggan",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_target, use_container_width=True)

    with tab4:
        st.markdown("#### Boxplot Distribusi Fitur Numerik (Analisis Outlier)")
        num_cols = ['age', 'total_spent', 'avg_order_value', 'lifetime_value',
                    'avg_session_time', 'pages_per_session', 'total_visits',
                    'satisfaction_score', 'marketing_spend_per_user']
        
        selected_boxplot_col = st.selectbox("Pilih Fitur Numerik untuk Boxplot", num_cols)
        
        fig_box = go.Figure(
            go.Box(
                y=df_raw[selected_boxplot_col].dropna(),
                name=selected_boxplot_col,
                boxpoints='outliers',
                marker_color='#3B82F6',
            )
        )
        fig_box.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("#### Cek Anomali Fitur Age")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.info(f"Jumlah baris Usia < 0 (Negatif): **{(df_raw['age'] < 0).sum()}**")
            st.info(f"Jumlah baris Usia > 80: **{(df_raw['age'] > 80).sum()}**")
        with col_a2:
            st.info(f"Usia Minimum di Dataset: **{df_raw['age'].min()}**")
            st.info(f"Usia Maksimum di Dataset: **{df_raw['age'].max()}**")

    with tab5:
        st.markdown("#### Heatmap Korelasi Fitur Numerik")
        
        num_cols_corr = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if 'customer_id' in num_cols_corr:
            num_cols_corr.remove('customer_id')
        corr_matrix = df_raw[num_cols_corr].corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            corr_matrix,
            annot=True, fmt='.2f',
            cmap='coolwarm', center=0,
            linewidths=0.5,
            annot_kws={'size': 6},
            ax=ax_corr
        )
        ax_corr.set_title('Heatmap Korelasi Fitur Numerik', fontsize=12, fontweight='bold')
        fig_corr.patch.set_alpha(0.0)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_corr)
        
        st.markdown("#### Top Korelasi Absolut Terhadap Churn")
        churn_corr = corr_matrix['churn'].drop('churn').abs().sort_values(ascending=False)
        st.dataframe(churn_corr.head(10).to_frame('Korelasi Absolut terhadap Churn'), use_container_width=True)

# =====================================================
# PAGE: PREDIKSI CHURN
# =====================================================
elif page == "🔮 Prediksi Churn":
    st.markdown('<div class="card"><div class="card-title">🧪 Pilih Contoh Data atau Isi Manual</div>', unsafe_allow_html=True)
    preset_name = st.selectbox(
        "Gunakan contoh cepat",
        list(PRESETS.keys()),
        index=2,
        help="Pilih contoh agar kamu tidak perlu mengisi dari nol. Nilainya masih bisa kamu ubah manual.",
    )
    preset = PRESETS[preset_name]
    st.markdown(
        '<div class="small-muted">Tips: pakai contoh “Pelanggan berisiko” untuk melihat output churn tinggi, atau “Pelanggan aktif/loyal” untuk output churn rendah.</div></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown('<div class="card"><div class="card-title">👤 1. Profil & Aktivitas Pelanggan</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider(FEATURE_INFO["age"]["label"], 17, 80, int(preset["age"]), help=FEATURE_INFO["age"]["simple"] + " " + FEATURE_INFO["age"]["why"])
            tenure_days = st.number_input(FEATURE_INFO["tenure_days"]["label"], min_value=0, max_value=5000, value=int(preset["tenure_days"]), step=1, help=FEATURE_INFO["tenure_days"]["simple"])
            days_since_last_purchase = st.number_input(FEATURE_INFO["days_since_last_purchase"]["label"], min_value=0, max_value=1200, value=int(preset["days_since_last_purchase"]), step=1, help=FEATURE_INFO["days_since_last_purchase"]["simple"])
            last_3_month_purchase_freq = st.number_input(FEATURE_INFO["last_3_month_purchase_freq"]["label"], min_value=0, max_value=200, value=int(preset["last_3_month_purchase_freq"]), step=1, help=FEATURE_INFO["last_3_month_purchase_freq"]["simple"])
        with c2:
            total_visits = st.number_input(FEATURE_INFO["total_visits"]["label"], min_value=0, max_value=1000, value=int(preset["total_visits"]), step=1, help=FEATURE_INFO["total_visits"]["simple"])
            avg_session_time = st.slider(FEATURE_INFO["avg_session_time"]["label"], 0.0, 120.0, float(preset["avg_session_time"]), step=0.1, help=FEATURE_INFO["avg_session_time"]["simple"])
            pages_per_session = st.slider(FEATURE_INFO["pages_per_session"]["label"], 1.0, 50.0, float(preset["pages_per_session"]), step=0.1, help=FEATURE_INFO["pages_per_session"]["simple"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">💰 2. Transaksi, Kepuasan, dan Keluhan</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            total_spent = st.number_input(FEATURE_INFO["total_spent"]["label"], min_value=0.0, max_value=100000.0, value=float(preset["total_spent"]), step=10.0, help=FEATURE_INFO["total_spent"]["simple"])
            avg_order_value = st.number_input(FEATURE_INFO["avg_order_value"]["label"], min_value=0.0, max_value=10000.0, value=float(preset["avg_order_value"]), step=5.0, help=FEATURE_INFO["avg_order_value"]["simple"])
            lifetime_value = st.number_input(FEATURE_INFO["lifetime_value"]["label"], min_value=0.0, max_value=200000.0, value=float(preset["lifetime_value"]), step=10.0, help=FEATURE_INFO["lifetime_value"]["simple"])
            marketing_spend_per_user = st.number_input(FEATURE_INFO["marketing_spend_per_user"]["label"], min_value=0.0, max_value=10000.0, value=float(preset["marketing_spend_per_user"]), step=1.0, help=FEATURE_INFO["marketing_spend_per_user"]["simple"])
        with c2:
            support_tickets = st.number_input(FEATURE_INFO["support_tickets"]["label"], min_value=0, max_value=50, value=int(preset["support_tickets"]), step=1, help=FEATURE_INFO["support_tickets"]["simple"])
            delivery_delay_days = st.number_input(FEATURE_INFO["delivery_delay_days"]["label"], min_value=0, max_value=30, value=int(preset["delivery_delay_days"]), step=1, help=FEATURE_INFO["delivery_delay_days"]["simple"])
            satisfaction_score = st.slider(FEATURE_INFO["satisfaction_score"]["label"], 1, 5, int(preset["satisfaction_score"]), help=FEATURE_INFO["satisfaction_score"]["example"])
            nps_score = st.slider(FEATURE_INFO["nps_score"]["label"], 0, 10, int(preset["nps_score"]), help=FEATURE_INFO["nps_score"]["example"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">📧 3. Respons Email Marketing</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            email_open_rate = st.slider(FEATURE_INFO["email_open_rate"]["label"], 0.0, 1.0, float(preset["email_open_rate"]), step=0.01, help=FEATURE_INFO["email_open_rate"]["example"])
        with c2:
            email_click_rate = st.slider(FEATURE_INFO["email_click_rate"]["label"], 0.0, 1.0, float(preset["email_click_rate"]), step=0.01, help=FEATURE_INFO["email_click_rate"]["example"])
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        input_data = {
            "age": age,
            "total_visits": total_visits,
            "avg_session_time": avg_session_time,
            "pages_per_session": pages_per_session,
            "email_open_rate": email_open_rate,
            "email_click_rate": email_click_rate,
            "total_spent": total_spent,
            "avg_order_value": avg_order_value,
            "support_tickets": support_tickets,
            "delivery_delay_days": delivery_delay_days,
            "satisfaction_score": satisfaction_score,
            "nps_score": nps_score,
            "marketing_spend_per_user": marketing_spend_per_user,
            "lifetime_value": lifetime_value,
            "last_3_month_purchase_freq": last_3_month_purchase_freq,
            "tenure_days": tenure_days,
            "days_since_last_purchase": days_since_last_purchase,
        }

        st.markdown('<div class="card"><div class="card-title">🎯 Hasil Prediksi</div>', unsafe_allow_html=True)
        run_prediction = st.button("🔮 Jalankan Prediksi", type="primary", use_container_width=True)

        if run_prediction:
            with st.spinner("Model sedang menganalisis data pelanggan..."):
                time.sleep(0.4)
                df_input = pd.DataFrame([input_data], columns=SCALE_COLS)
                scaled = scaler.transform(df_input)
                df_scaled = pd.DataFrame(scaled, columns=SCALE_COLS)
                df_final = df_scaled[selected_features]

                pred = int(model.predict(df_final)[0])
                proba = float(model.predict_proba(df_final)[0][1])
                percentage = proba * 100
                label, css_class, meaning = risk_label(percentage)

            st.plotly_chart(make_gauge(percentage), use_container_width=True)
            st.markdown(
                f'<div class="{css_class}">Risiko Churn: {label}<br><span style="font-weight:500;">{meaning}</span></div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Kenapa hasilnya bisa begitu?")
            for signal in explain_input_signals(input_data):
                st.write(f"- {signal}")

            st.markdown("#### Rekomendasi tindakan")
            if percentage >= 60:
                st.write("- Prioritaskan pelanggan ini untuk program retensi.")
                st.write("- Berikan voucher, diskon, atau benefit personal.")
                st.write("- Hubungi pelanggan untuk menyelesaikan keluhan dan meminta feedback.")
            elif percentage >= 30:
                st.write("- Pantau aktivitas pelanggan dalam beberapa minggu ke depan.")
                st.write("- Kirim kampanye engagement yang relevan.")
                st.write("- Perbaiki faktor yang terlihat lemah, misalnya kepuasan, keluhan, atau aktivitas belanja.")
            else:
                st.write("- Pertahankan engagement pelanggan.")
                st.write("- Tawarkan upselling/cross-selling karena pelanggan relatif loyal.")
                st.write("- Ajak mengikuti referral atau loyalty program.")

            st.session_state.prediction_history.insert(
                0,
                {
                    "Waktu": datetime.now().strftime("%d %b %Y %H:%M"),
                    "Risiko": label,
                    "Probabilitas Churn": f"{percentage:.1f}%",
                    "Usia": age,
                    "Kepuasan": satisfaction_score,
                    "Support Tickets": support_tickets,
                    "Hari Sejak Beli": days_since_last_purchase,
                },
            )
        else:
            st.info("Isi data di sebelah kiri, lalu klik tombol prediksi.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🔎 Faktor Paling Berpengaruh di Model</div>', unsafe_allow_html=True)
        fig_top = top_feature_chart(model, selected_features)
        if fig_top is not None:
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.caption("Model tidak memiliki feature_importances_.")
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PAGE: PANDUAN FITUR
# =====================================================
elif page == "📘 Panduan Fitur":
    st.markdown('<div class="card"><div class="card-title">📘 Arti Fitur yang Dipakai Model</div>', unsafe_allow_html=True)
    st.write("Halaman ini menjelaskan input yang ada di form prediksi. Kamu bisa pakai bagian ini untuk presentasi juga.")
    st.markdown("</div>", unsafe_allow_html=True)

    for feature in SCALE_COLS:
        info = FEATURE_INFO[feature]
        with st.expander(f"{info['label']} ({feature})"):
            st.write(f"**Artinya:** {info['simple']}")
            st.write(f"**Kenapa penting:** {info['why']}")
            st.write(f"**Contoh input:** {info['example']}")

# =====================================================
# PAGE: RIWAYAT ANALISIS
# =====================================================
elif page == "📈 Riwayat Analisis":
    st.markdown('<div class="card"><div class="card-title">📈 Riwayat Prediksi</div>', unsafe_allow_html=True)
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
        if st.button("Hapus Riwayat", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("Belum ada riwayat. Jalankan prediksi dulu di menu Prediksi.")
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PAGE: INFO MODEL
# =====================================================
elif page == "⚙️ Info Model":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><div class="card-title">🤖 Informasi Model</div>', unsafe_allow_html=True)
        st.write("**Model:** Decision Tree Classifier")
        st.write("**Dataset:** Sales & Marketing Customer Dataset")
        st.write("**Jumlah data:** 15.000 records")
        st.write("**Target:** churn")
        st.write("**File model dimuat dari:**")
        st.code(loaded_from_path)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">📁 Status File Deployment</div>', unsafe_allow_html=True)
        st.success("best_model.pkl ditemukan")
        st.success("scaler.pkl ditemukan")
        st.success("selected_features.pkl ditemukan")
        st.info("Pastikan requirements.txt berisi: streamlit, pandas, numpy, scikit-learn, joblib, plotly")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🧩 Fitur yang Dipakai Saat Prediksi</div>', unsafe_allow_html=True)
    chips = "".join([f'<span class="feature-chip">{FEATURE_INFO.get(f, {}).get("label", f)}</span>' for f in selected_features])
    st.markdown(chips, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center; color:#64748B; font-size:0.85rem; padding:24px 0;">
        ChurnGuard v2.0 | UAS Bengkel Koding Data Science | Streamlit Deployment Ready
    </div>
    """,
    unsafe_allow_html=True,
)
