# 🔍 UAS Bengkel Koding Data Science
## Customer Churn Prediction - Sales & Marketing Dataset

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red)

---

## 📌 Deskripsi Proyek
Proyek ini bertujuan membangun model prediksi **customer churn** menggunakan 
dataset Sales & Marketing (15.000 records, 30 fitur). Model terbaik akan 
di-deploy menggunakan Streamlit Cloud.

---

## 📁 Struktur Repository
uas-churn-prediction/
├── data/
│   └── Sales_-_Marketing_customer_dataset.csv
├── notebook/
│   └── UAS_Churn_Prediction.ipynb
├── app/
│   └── app.py
├── requirements.txt
└── README.md

---

## 🧪 Eksperimen Model
| Kategori | Model |
|---|---|
| Konvensional | Decision Tree |
| Ensemble Bagging | Random Forest |
| Ensemble Voting | RF + GBT + LR |

### 3 Skenario Eksperimen:
1. **Direct Modeling** — tanpa preprocessing
2. **Preprocessing** — handling missing value, encoding, scaling
3. **Hyperparameter Tuning** — optimasi model terbaik

---

## 📊 Progress
- [x] EDA (Exploratory Data Analysis)
- [x] Direct Modeling
- [x] Preprocessing & Modeling
- [ ] Hyperparameter Tuning
- [ ] Deployment ke Streamlit Cloud

---

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, Matplotlib, Seaborn
- Streamlit
