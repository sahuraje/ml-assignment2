import streamlit as st
import pandas as pd
import pickle, json
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, matthews_corrcoef,
                              confusion_matrix, classification_report)

st.title("Telco Customer Churn — Classification Model Comparison")
st.caption("BITS M.Tech AIML — ML Assignment 2 | Rajendra Sahu")

with open("model/results.json", "r") as f:  # load saved metrics from training
    all_results = json.load(f)

st.subheader("Model Accuracy Comparison (Test Set)")
accuracy_df = pd.DataFrame({
    "Model": list(all_results.keys()),
    "Accuracy": [all_results[m]["Accuracy"] for m in all_results]
}).sort_values("Accuracy", ascending=False)  # highest accuracy first

st.bar_chart(accuracy_df.set_index("Model"))  # quick visual before upload

st.divider()

model_names = ["Random Forest", "Logistic Regression", "Naive Bayes", "kNN", "Decision Tree"]  # my own order
selected = st.selectbox("Select Model for Detailed Evaluation", model_names)

uploaded = st.file_uploader("Upload test data (CSV)", type="csv")  # only test data, not full training set

if uploaded:
    data = pd.read_csv(uploaded)
    y_true = data["Churn"]  # actual labels
    X = data.drop("Churn", axis=1)  # features only

    with open(f"model/{selected.replace(' ', '_')}.pkl", "rb") as f:  # load chosen model
        model = pickle.load(f)

    if selected in ["Logistic Regression", "kNN"]:  # these need scaled input
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        X_input = scaler.transform(X)
    else:  # tree/probability based models, no scaling needed
        X_input = X

    preds = model.predict(X_input)
    probs = model.predict_proba(X_input)[:, 1]  # needed for AUC

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
    st.write(confusion_matrix(y_true, preds))  # rows=actual, cols=predicted

    st.subheader("Classification Report")
    report_dict = classification_report(y_true, preds, output_dict=True)  # get as dict instead of plain text
    report_df = pd.DataFrame(report_dict).transpose().round(3)  # convert to table, round numbers
    st.dataframe(report_df, use_container_width=True)  # nice scrollable/formatted table
