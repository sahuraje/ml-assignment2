import streamlit as st
import pandas as pd
import pickle, json
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, matthews_corrcoef,
                              confusion_matrix, classification_report)

st.title("Classification Model Comparison — Telco Churn")

model_names = ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"]
selected = st.selectbox("Select Model", model_names)

uploaded = st.file_uploader("Upload test data (CSV)", type="csv")

if uploaded:
    data = pd.read_csv(uploaded)
    y_true = data["Churn"]
    X = data.drop("Churn", axis=1)

    with open(f"model/{selected.replace(' ', '_')}.pkl", "rb") as f:
        model = pickle.load(f)

    if selected in ["Logistic Regression", "kNN"]:
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        X_input = scaler.transform(X)
    else:
        X_input = X

    preds = model.predict(X_input)
    probs = model.predict_proba(X_input)[:, 1]

    st.subheader(f"Metrics — {selected}")
    st.write({
        "Accuracy": round(accuracy_score(y_true, preds), 4),
        "AUC": round(roc_auc_score(y_true, probs), 4),
        "Precision": round(precision_score(y_true, preds), 4),
        "Recall": round(recall_score(y_true, preds), 4),
        "F1": round(f1_score(y_true, preds), 4),
        "MCC": round(matthews_corrcoef(y_true, preds), 4)
    })

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_true, preds))

    st.subheader("Classification Report")
    st.text(classification_report(y_true, preds))
