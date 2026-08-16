import streamlit as st
import pandas as pd
import pickle, json
import altair as alt
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

# Altair chart with value labels on top of each bar, and color legend by model
bar_chart = alt.Chart(accuracy_df).mark_bar().encode(
    x=alt.X("Model", sort=None, title="Model"),
    y=alt.Y("Accuracy", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
    color=alt.Color("Model", legend=alt.Legend(title="Model"))  # adds legend
)

text_labels = alt.Chart(accuracy_df).mark_text(
    align="center", baseline="bottom", dy=-3  # dy moves label slightly above bar
).encode(
    x=alt.X("Model", sort=None),
    y="Accuracy",
    text=alt.Text("Accuracy", format=".4f")  # show value rounded to 4 decimals
)

st.altair_chart((bar_chart + text_labels).properties(height=400), use_container_width=True)  # combine bars + labels

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
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        "Value": [
            round(accuracy_score(y_true, preds), 4),
            round(roc_auc_score(y_true, probs), 4),
            round(precision_score(y_true, preds), 4),
            round(recall_score(y_true, preds), 4),
            round(f1_score(y_true, preds), 4),
            round(matthews_corrcoef(y_true, preds), 4)
        ]
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)  # clean table, no index column

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, preds)
    cm_df = pd.DataFrame(cm,
                          index=["Actual: No Churn", "Actual: Churn"],
                          columns=["Predicted: No Churn", "Predicted: Churn"])
    st.dataframe(cm_df, use_container_width=True)   # rows=actual, cols=predicted

    st.subheader("Classification Report")
    report_dict = classification_report(y_true, preds, output_dict=True)  # get as dict instead of plain text
    report_df = pd.DataFrame(report_dict).transpose().round(3)  # convert to table, round numbers
    st.dataframe(report_df, use_container_width=True)  # formatted table