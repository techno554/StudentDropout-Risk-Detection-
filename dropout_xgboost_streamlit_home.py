# dropout_xgboost_streamlit_uploader.py (versi dengan Home dan Grafik)

import pandas as pd
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Streamlit page config
st.set_page_config(page_title="Dropout Risk Dashboard", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard"])

# Home Page
if page == "Home":
    st.title("🎓 Student Dropout Risk Detection App")
    st.markdown("""
    Selamat datang di aplikasi **Prediksi Risiko Dropout Mahasiswa**.

    Aplikasi ini memanfaatkan:
    - **XGBoost** untuk klasifikasi risiko dropout,
    - **SHAP** untuk menjelaskan faktor penyebab,
    - dan **Streamlit** untuk antarmuka interaktif.

    **Silakan navigasi ke halaman *Dashboard* di sidebar untuk memulai.**
    """)

# Dashboard Page
elif page == "Dashboard":
    st.title("📊 Dropout Prediction Dashboard")

    uploaded_file = st.file_uploader("📤 Upload Dataset CSV", type=["csv"])

    @st.cache_data
    def load_data(uploaded_file):
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        df = df.dropna(axis=1, how='all')
        df['Dropout'] = np.where((df['IPK'] < 2.5) & (df['SKS'] < 80), 1, 0)
        return df

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        st.subheader("📄 Dataset")
        st.dataframe(df)

        features = ['SKS', 'IPK', 'Ikut Organisasi', 'Ikut UKM', 'Status Beasiswa', 'Penghasilan']
        df_model = df[features + ['Dropout']].copy()

        for col in ['Ikut Organisasi', 'Ikut UKM', 'Status Beasiswa', 'Penghasilan']:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))

        X = df_model.drop('Dropout', axis=1)
        y = df_model['Dropout']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Evaluation
        st.subheader("📈 Model Evaluation")
        st.json(report)

        # Grafik Dropout
        st.subheader("📊 Grafik Jumlah Dropout vs Tidak Dropout")
        fig_bar, ax = plt.subplots()
        sns.countplot(data=df, x='Dropout', ax=ax)
        ax.set_xticklabels(['Tidak Dropout', 'Dropout'])
        ax.set_ylabel("Jumlah Mahasiswa")
        ax.set_xlabel("Status")
        st.pyplot(fig_bar)

        # SHAP Explainability
        st.subheader("🔍 SHAP Feature Importance")
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        fig_summary = shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        st.pyplot(bbox_inches='tight', clear_figure=True)

        # Individual prediction
        st.subheader("👤 Prediksi Mahasiswa Individual")
        idx = st.slider("Pilih Indeks Mahasiswa", 0, len(X) - 1, 0)
        student = X.iloc[[idx]]
        student_pred = model.predict(student)[0]
        student_prob = model.predict_proba(student)[0][1]

        st.write(f"### Prediksi: {'Dropout' if student_pred == 1 else 'Tidak Dropout'}")
        st.write(f"Probabilitas Dropout: {student_prob:.2f}")

        with st.expander("📌 Penjelasan SHAP untuk Mahasiswa Ini"):
            shap.plots.waterfall(shap_values[idx], show=False)
            st.pyplot(bbox_inches='tight', clear_figure=True)

        st.markdown("---")
        st.caption("Built with Streamlit, XGBoost, and SHAP for educational analytics.")
    else:
        st.warning("⚠️ Silakan upload file CSV terlebih dahulu.")