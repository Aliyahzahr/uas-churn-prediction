import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Set page configuration
st.set_page_config(
    page_title="ChurnGuard | Analytics Dashboard",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeksi CSS Kustom untuk tampilan premium & Neumorphic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background-color: #F3F4F9 !important;
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #1E293B;
    }

    /* Styling Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #EAECEF !important;
        box-shadow: 2px 0 15px rgba(166, 173, 201, 0.08) !important;
    }
    
    .sidebar-logo {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: #6F52ED;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Card Desain Utama */
    .dashboard-card {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(166, 173, 201, 0.12);
        border: 1px solid #EAECEF;
        margin-bottom: 24px;
    }
    
    .dashboard-card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 2px solid #F3F4F9;
        padding-bottom: 10px;
    }

    /* Welcome Banner Card */
    .welcome-card {
        background-color: #FFFFFF;
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(166, 173, 201, 0.12);
        border: 1px solid #EAECEF;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .welcome-text {
        font-size: 1.1rem;
        color: #718096;
        margin-bottom: 4px;
    }
    
    .welcome-user {
        font-size: 2rem;
        font-weight: 800;
        color: #1E293B;
        margin: 0;
        line-height: 1.2;
    }
    
    .welcome-time {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #6F52ED;
        margin-top: 10px;
        margin-bottom: 0;
    }

    /* Result Card Banners */
    .result-card-churn {
        background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
        border-left: 6px solid #E53E3E;
        padding: 20px;
        border-radius: 16px;
        color: #9B2C2C;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(229, 62, 62, 0.08);
    }
    
    .result-card-loyal {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border-left: 6px solid #16A34A;
        padding: 20px;
        border-radius: 16px;
        color: #14532D;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(22, 163, 74, 0.08);
    }
    
    .result-title {
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    
    /* Customize Streamlit Buttons & Radio */
    div.stButton > button:first-child {
        background-color: #6F52ED !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 20px rgba(111, 82, 237, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 24px rgba(111, 82, 237, 0.35) !important;
    }
    
    /* Radio styling for navigation block */
    div[data-testid="stSidebarNav"] {
        display: none;
    }
    
    .stRadio > div {
        background: transparent !important;
        padding: 0 !important;
        gap: 8px !important;
    }
    
    .stRadio label {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        color: #4A5568 !important;
    }
    
    /* Footer styling */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        color: #718096;
        font-size: 0.8rem;
        border-top: 1px solid #E2E8F0;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model, scaler, dan selected features secara dinamis (dengan fallback path)
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    paths_to_try = [
        os.path.join(current_dir, '..', 'notebook', 'model'), 
        os.path.join(current_dir, 'model'),                    
        os.path.join(current_dir, '..', 'model'),               
        './model',
        '.'
    ]
    
    model_name = 'best_model.pkl'
    scaler_name = 'scaler.pkl'
    features_name = 'selected_features.pkl'
    
    for path in paths_to_try:
        model_path = os.path.join(path, model_name)
        scaler_path = os.path.join(path, scaler_name)
        features_path = os.path.join(path, features_name)
        
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            selected_features = joblib.load(features_path)
            return model, scaler, selected_features, path
            
    raise FileNotFoundError("Gagal memuat model, scaler, atau selected_features dari semua jalur alternatif.")

try:
    model, scaler, selected_features, loaded_from_path = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Gagal memuat aset model: {e}")

# --- SIDEBAR NAVIGASI INTERAKTIF ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛫 ChurnGuard</div>', unsafe_allow_html=True)
    st.markdown("### 📊 Menu Utama")
    
    menu_selection = st.radio(
        "Navigasi Halaman",
        options=["🏠 Dashboard", "📈 Riwayat Analisis", "⚙️ Konfigurasi"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📝 Detail Mahasiswa")
    st.info("Aliyah Zahratu Rizqi\nNIM: A11.2023.15294")

# --- MAIN PAGE CONTENT ---

if assets_loaded:
    # Waktu dinamis
    now = datetime.now()
    current_hour = now.hour
    if current_hour < 12:
        greeting = "Selamat Pagi"
    elif current_hour < 17:
        greeting = "Selamat Siang"
    else:
        greeting = "Selamat Malam"
        
    time_str = now.strftime("%I:%M %p").lower()
    date_str = now.strftime("%A, %d %B %Y")
    
    # 1. Welcome Header Card
    col_text, col_img = st.columns([7, 5])
    with col_text:
        st.markdown(f"""
        <div class="welcome-card" style="height: 100%;">
            <div style="width: 100%;">
                <div class="welcome-text">{greeting}, Aliyah Zahratu Rizqi! Have a fruitful day ahead.</div>
                <h2 class="welcome-user">Analytics Command Center</h2>
                <div class="welcome-time">{time_str}</div>
                <div style="color: #718096; margin-top: 5px; font-weight: 500;">📅 {date_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_img:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        banner_img_path = os.path.join(current_dir, 'churn_guard_banner.png')
        if os.path.exists(banner_img_path):
            st.image(banner_img_path, use_container_width=True)
        else:
            st.markdown("""
            <div class="welcome-card" style="height: 100%; display: flex; align-items: center; justify-content: center;">
                <span style="color: #A0AEC0;">Illustration Placeholder</span>
            </div>
            """, unsafe_allow_html=True)

    # PAGE 1: DASHBOARD OVERVIEW
    if menu_selection == "🏠 Dashboard":
        # Definisikan scale_cols sesuai urutan fitting scaler
        scale_cols = [
            'age', 'total_visits', 'avg_session_time', 'pages_per_session',
            'email_open_rate', 'email_click_rate', 'total_spent',
            'avg_order_value', 'support_tickets', 'delivery_delay_days',
            'satisfaction_score', 'nps_score', 'marketing_spend_per_user',
            'lifetime_value', 'last_3_month_purchase_freq',
            'tenure_days', 'days_since_last_purchase'
        ]

        # Grid Utama (Kiri: Inputs, Kanan: Hasil & Rekomendasi)
        col_left, col_right = st.columns([7, 5])
        
        with col_left:
            # Card 1: Profil & Riwayat Pelanggan
            st.markdown("""
            <div class="dashboard-card">
                <div class="dashboard-card-title">👤 Profil & Riwayat Pelanggan</div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Usia Pelanggan", 17, 80, 35, help="Usia pelanggan (17-80 tahun)")
                tenure_days = st.number_input("Tenure (Hari)", min_value=0, max_value=5000, value=660, step=1, help="Berapa hari pelanggan sudah terdaftar")
            with c2:
                days_since_last_purchase = st.number_input("Hari Sejak Pembelian Terakhir", min_value=0, max_value=365*3, value=397, step=1, help="Berapa hari sejak terakhir kali pelanggan melakukan pembelian")
                last_3_month_purchase_freq = st.number_input("Frekuensi Pembelian (3 Bulan Terakhir)", min_value=0, max_value=200, value=3, step=1, help="Frekuensi belanja pelanggan dalam 3 bulan terakhir")
            st.markdown("</div>", unsafe_allow_html=True)

            # Card 2: Aktivitas Penggunaan
            st.markdown("""
            <div class="dashboard-card">
                <div class="dashboard-card-title">📱 Aktivitas Penggunaan</div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                total_visits = st.number_input("Total Kunjungan Aplikasi", min_value=0, max_value=1000, value=12, step=1, help="Total kunjungan pelanggan ke aplikasi")
                avg_session_time = st.slider("Rata-rata Durasi Sesi (Menit)", 0.0, 120.0, 8.0, step=0.1, help="Rata-rata waktu yang dihabiskan pelanggan per kunjungan (menit)")
                pages_per_session = st.slider("Halaman per Sesi", 1.0, 50.0, 4.0, step=0.1, help="Rata-rata halaman aplikasi yang dibuka per kunjungan")
            with c2:
                email_open_rate = st.slider("Email Open Rate", 0.0, 1.0, 0.3, step=0.01, help="Rasio email dari perusahaan yang dibuka oleh pelanggan (0-1)")
                email_click_rate = st.slider("Email Click Rate", 0.0, 1.0, 0.1, step=0.01, help="Rasio tautan email yang diklik oleh pelanggan (0-1)")
                support_tickets = st.number_input("Jumlah Tiket Pengaduan (Support)", min_value=0, max_value=50, value=2, step=1, help="Jumlah pengaduan/tiket keluhan pelanggan")
            st.markdown("</div>", unsafe_allow_html=True)

            # Card 3: Keuangan & Kepuasan
            st.markdown("""
            <div class="dashboard-card">
                <div class="dashboard-card-title">💰 Keuangan & Kepuasan</div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                total_spent = st.number_input("Total Pengeluaran (Spent)", min_value=0.0, max_value=100000.0, value=350.0, step=10.0, help="Total uang yang dikeluarkan pelanggan")
                avg_order_value = st.number_input("Rata-rata Nilai Pesanan (AOV)", min_value=0.0, max_value=10000.0, value=50.0, step=5.0, help="Rata-rata nilai belanja per transaksi")
                lifetime_value = st.number_input("Customer Lifetime Value (CLV)", min_value=0.0, max_value=200000.0, value=700.0, step=10.0, help="Estimasi nilai total keuntungan dari pelanggan selama berlangganan")
                marketing_spend_per_user = st.number_input("Marketing Spend per User", min_value=0.0, max_value=10000.0, value=15.0, step=1.0, help="Biaya pemasaran yang dialokasikan untuk pelanggan")
            with c2:
                delivery_delay_days = st.number_input("Hari Keterlambatan Pengiriman", min_value=0, max_value=30, value=2, step=1, help="Rata-rata hari keterlambatan pengiriman barang")
                satisfaction_score = st.slider("Skor Kepuasan (Satisfaction)", 1, 5, 4, help="Skor kepuasan 1-5, dimana 5 = sangat puas")
                nps_score = st.slider("NPS Score (Net Promoter Score)", 0, 10, 8, help="Seberapa besar kemungkinan pelanggan merekomendasikan layanan (0-10)")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            # Card Panel Kanan: Command Desk & Input Validation Warning
            st.markdown("""
            <div class="dashboard-card" style="text-align: center;">
                <div class="dashboard-card-title">🚀 Command Desk</div>
            """, unsafe_allow_html=True)
            
            if satisfaction_score <= 2 and support_tickets >= 5 and total_spent < 100:
                st.warning("⚠️ Data menunjukkan tanda-tanda risiko churn tinggi. Pastikan data yang dimasukkan sudah benar.")
            else:
                st.markdown("""
                <p style="color: #718096; font-size: 0.95rem; margin-bottom: 20px;">
                    Klik tombol di bawah ini untuk memicu analisis model secara realtime terhadap data masukan.
                </p>
                """, unsafe_allow_html=True)
                
            btn_prediksi = st.button("Jalankan Prediksi Churn", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Menampilkan output prediksi
            if btn_prediksi:
                with st.spinner("Menganalisis data pelanggan..."):
                    time.sleep(0.6)
                    
                    numeric_inputs = {
                        'age': age,
                        'total_visits': total_visits,
                        'avg_session_time': avg_session_time,
                        'pages_per_session': pages_per_session,
                        'email_open_rate': email_open_rate,
                        'email_click_rate': email_click_rate,
                        'total_spent': total_spent,
                        'avg_order_value': avg_order_value,
                        'support_tickets': support_tickets,
                        'delivery_delay_days': delivery_delay_days,
                        'satisfaction_score': satisfaction_score,
                        'nps_score': nps_score,
                        'marketing_spend_per_user': marketing_spend_per_user,
                        'lifetime_value': lifetime_value,
                        'last_3_month_purchase_freq': last_3_month_purchase_freq,
                        'tenure_days': tenure_days,
                        'days_since_last_purchase': days_since_last_purchase
                    }
                    
                    # Scaling & Ordering DataFrame
                    df_numeric = pd.DataFrame([numeric_inputs], columns=scale_cols)
                    scaled_numeric = scaler.transform(df_numeric)[0]
                    df_scaled = pd.DataFrame([scaled_numeric], columns=scale_cols)
                    df_final = df_scaled[selected_features]
                    
                    # Prediksi
                    pred = model.predict(df_final)[0]
                    proba = model.predict_proba(df_final)[0][1]
                    
                    # Log to prediction history dynamically
                    waktu_pred = datetime.now().strftime("%d %b %Y %H:%M")
                    status_risiko = 'RISIKO TINGGI' if pred == 1 else 'RISIKO RENDAH'
                    st.session_state.prediction_history.insert(0, {
                        'Waktu Prediksi': waktu_pred,
                        'Usia': int(age),
                        'Tenure (Hari)': int(tenure_days),
                        'Total Spent ($)': float(total_spent),
                        'Skor Kepuasan': int(satisfaction_score),
                        'Persentase Churn': f"{proba * 100:.1f}%",
                        'Status Risiko': status_risiko
                    })
                    
                st.markdown("""
                <div class="dashboard-card">
                    <div class="dashboard-card-title">📊 Hasil Analisis Prediksi</div>
                """, unsafe_allow_html=True)
                
                # a) Probability gauge chart
                percentage = proba * 100
                if percentage <= 30:
                    color = "#16A34A"  # Green
                    level = "LOW RISK"
                elif percentage <= 60:
                    color = "#EAB308"  # Yellow
                    level = "MEDIUM RISK"
                else:
                    color = "#E53E3E"  # Red
                    level = "HIGH RISK"
                    
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=percentage,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    number={'suffix': "%", 'font': {'size': 36, 'family': 'Outfit', 'color': color}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#718096"},
                        'bar': {'color': color},
                        'bgcolor': "#F3F4F9",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 30], 'color': '#DCFCE7'},
                            {'range': [30, 60], 'color': '#FEF9C3'},
                            {'range': [60, 100], 'color': '#FEE2E2'}
                        ],
                    }
                ))
                fig.update_layout(
                    height=180,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # b) Risk badge & c) Confidence percentage text
                if pred == 1:
                    st.markdown(f"""
                    <div class="result-card-churn" style="margin-top: 10px;">
                        <div class="result-title">⚠️ RISIKO TINGGI (CHURN)</div>
                        Kemungkinan pelanggan berhenti berlangganan tergolong tinggi dengan tingkat keyakinan model sebesar <b>{percentage:.1f}%</b>.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-loyal" style="margin-top: 10px;">
                        <div class="result-title">✅ RISIKO RENDAH (LOYAL/AKTIF)</div>
                        Pelanggan diprediksi akan tetap setia menggunakan produk dengan tingkat keyakinan model sebesar <b>{100 - percentage:.1f}%</b>.
                    </div>
                    """, unsafe_allow_html=True)
                    
                # d) Top 5 feature importance chart
                importances = model.feature_importances_
                df_imp = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(5)
                
                clean_labels = {
                    'total_spent': 'Total Spent',
                    'satisfaction_score': 'Sat Score',
                    'support_tickets': 'Support Tickets',
                    'pages_per_session': 'Pages/Sess',
                    'marketing_spend_per_user': 'Mkt Spend',
                    'lifetime_value': 'CLV',
                    'avg_session_time': 'Sess Time',
                    'avg_order_value': 'AOV',
                    'days_since_last_purchase': 'Recency',
                    'tenure_days': 'Tenure',
                    'email_open_rate': 'Email Open',
                    'email_click_rate': 'Email Click',
                    'age': 'Age',
                    'total_visits': 'Visits',
                    'last_3_month_purchase_freq': 'Freq (3m)',
                    'nps_score': 'NPS',
                    'delivery_delay_days': 'Del Delay'
                }
                df_imp['UI_Label'] = df_imp['Feature'].map(clean_labels)
                
                fig_imp, ax_imp = plt.subplots(figsize=(5, 2.5))
                sns.barplot(
                    x='Importance',
                    y='UI_Label',
                    data=df_imp,
                    ax=ax_imp,
                    palette=sns.color_palette("Purples_r", 5),
                    hue='UI_Label',
                    legend=False
                )
                ax_imp.set_title("Faktor Utama Penentu Prediksi", fontsize=9, fontweight='bold', fontfamily='Plus Jakarta Sans')
                ax_imp.set_xlabel("Importance Score", fontsize=7)
                ax_imp.set_ylabel("", fontsize=7)
                ax_imp.tick_params(axis='both', which='major', labelsize=7)
                
                fig_imp.patch.set_alpha(0.0)
                ax_imp.patch.set_alpha(0.0)
                sns.despine(ax=ax_imp, left=True, bottom=True)
                plt.tight_layout()
                st.pyplot(fig_imp)
                
                # e) Recommendations list
                st.markdown("#### 💡 Tindakan Rekomendasi:")
                if pred == 1:
                    st.markdown("""
                    - **Diskon & Reward Loyalitas:** Berikan penawaran harga personal atau gratis ongkir pada pesanan berikutnya.
                    - **Evaluasi Layanan Pelanggan:** Segera selesaikan semua tiket pengaduan aktif dan lakukan panggilan feedback.
                    - **Keterlibatan Kembali:** Kirimkan kampanye email yang ditargetkan khusus untuk menarik perhatian kembali.
                    """)
                else:
                    st.markdown("""
                    - **Upselling & Cross-selling:** Tawarkan paket berlangganan tahunan dengan harga diskon premium.
                    - **Program Advokasi Pelanggan:** Undang untuk bergabung dalam program referral guna membagikan pengalaman positif mereka.
                    """)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="dashboard-card" style="text-align: center; color: #A0AEC0; padding: 40px 20px;">
                    <span style="font-size: 3rem;">📥</span>
                    <p style="margin-top: 10px; font-weight: 500;">Silakan klik tombol di atas untuk menjalankan analisis.</p>
                </div>
                """, unsafe_allow_html=True)

    # PAGE 2: DYNAMIC HISTORY LOG
    elif menu_selection == "📈 Riwayat Analisis":
        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-card-title">📈 Log Riwayat Analisis Churn</div>
            <p style="color: #718096; margin-bottom: 20px;">
                Daftar analisis prediksi churn pelanggan terbaru yang tersimpan dalam sistem lokal.
            </p>
        """, unsafe_allow_html=True)
        
        if not st.session_state.prediction_history:
            st.info("ℹ️ Belum ada riwayat analisis. Silakan lakukan prediksi baru pada halaman Dashboard.")
        else:
            history_data = pd.DataFrame(st.session_state.prediction_history)
            
            # Function to color risk status
            def color_risk(val):
                color = '#E53E3E' if val == 'RISIKO TINGGI' else '#16A34A'
                return f'color: {color}; font-weight: bold;'
                
            st.dataframe(
                history_data.style.map(color_risk, subset=['Status Risiko']),
                use_container_width=True
            )
            
            # Add a clear history button inside the card for premium user experience
            if st.button("Hapus Semua Riwayat"):
                st.session_state.prediction_history = []
                st.rerun()
                
        st.markdown("</div>", unsafe_allow_html=True)

    # PAGE 3: MODEL CONFIGURATION
    elif menu_selection == "⚙️ Konfigurasi":
        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-card-title">⚙️ Konfigurasi Model & Sistem</div>
        """, unsafe_allow_html=True)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### 🤖 Informasi Model Klasifikasi")
            st.write(f"**Tipe Model:** Decision Tree Classifier (Tuned)")
            st.write(f"**Jalur Pemuatan Aset:** `{loaded_from_path}`")
            st.write(f"**Metrik Performa:** Accuracy 86% | F1-Score 0.54")
            st.write(f"**Jumlah Fitur Latih:** 17 Fitur Numerik Utama")
        with col_c2:
            st.markdown("#### 📁 Status File Aset")
            st.success("✓ Model (`best_model.pkl`) berhasil dimuat")
            st.success("✓ Scaler (`scaler.pkl`) berhasil dimuat")
            st.success("✓ Features List (`selected_features.pkl`) berhasil dimuat")
            
        st.markdown("---")
        st.markdown("#### 📊 Daftar Parameter Fitur Terpilih (17 Fitur)")
        st.write(", ".join(selected_features))
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("Silakan ekspor model terlebih dahulu dari Jupyter Notebook.")

# --- FOOTER APLIKASI ---
st.markdown("""
<div class="app-footer">
    ChurnGuard v1.0 | UAS Bengkel Koding Data Science | Model: Decision Tree | Accuracy: 86% | F1-Score: 0.54 | Dataset: Sales & Marketing (15.000 records)
</div>
""", unsafe_allow_html=True)
