# dropout_xgboost_streamlit.py

import pandas as pd
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Streamlit page config
st.set_page_config(layout="wide")
st.title("🎓 Student Dropout Risk Detection Dashboard")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("1000 Data Fix (1).csv", encoding='ISO-8859-1')
    df = df.dropna(axis=1, how='all')
    df['Dropout'] = np.where((df['IPK'] < 2.5) & (df['SKS'] < 80), 1, 0)
    return df

df = load_data()

# Display dataset
with st.expander("📄 View Dataset"):
    st.dataframe(df)

# Feature selection
features = ['SKS', 'IPK', 'Ikut Organisasi', 'Ikut UKM', 'Status Beasiswa', 'Penghasilan']
df_model = df[features + ['Dropout']].copy()

# Encode categorical columns
for col in ['Ikut Organisasi', 'Ikut UKM', 'Status Beasiswa', 'Penghasilan']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# Train-test split
X = df_model.drop('Dropout', axis=1)
y = df_model['Dropout']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Show metrics
st.subheader("📊 Model Evaluation")
st.write("**Classification Report:**")
st.json(report)

# SHAP Feature Importance
st.subheader("🔍 SHAP Feature Importance")
explainer = shap.Explainer(model)
shap_values = explainer(X)

fig_summary = shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
st.pyplot(bbox_inches='tight', clear_figure=True)

# Individual prediction
st.subheader("👤 Predict Individual Student")
idx = st.slider("Select Student Index", 0, len(X) - 1, 0)
student = X.iloc[[idx]]
student_pred = model.predict(student)[0]
student_prob = model.predict_proba(student)[0][1]

st.write(f"### Prediction: {'Dropout' if student_pred == 1 else 'Not Dropout'}")
st.write(f"Probability of Dropout: {student_prob:.2f}")

with st.expander("📌 SHAP Explanation for This Student"):
    shap.plots.waterfall(shap_values[idx], show=False)
    st.pyplot(bbox_inches='tight', clear_figure=True)

# Footer
st.markdown("---")
st.caption("Built with Streamlit, XGBoost, and SHAP for educational analytics.")