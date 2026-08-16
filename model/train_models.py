import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, matthews_corrcoef)
import pickle, json

df = pd.read_csv("../WA_Fn-UseC_-Telco-Customer-Churn.csv")  # load raw dataset
df.drop("customerID", axis=1, inplace=True)  # not a useful feature, just an ID
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # some blanks in this column, force to number
df.dropna(inplace=True)  # drop rows where TotalCharges couldn't convert

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:  # encode all text columns to numbers
    df[col] = le.fit_transform(df[col])

X = df.drop("Churn", axis=1)  # features
y = df["Churn"]  # target label (0 = no churn, 1 = churn)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)  # 80/20 split, stratify keeps churn ratio same in both sets

scaler = StandardScaler()  # needed for Logistic Regression and kNN
X_train_s = scaler.fit_transform(X_train)  # fit only on train data to avoid data leakage
X_test_s = scaler.transform(X_test)  # transform test using same scaler

test_df = X_test.copy()
test_df["Churn"] = y_test
test_df.to_csv("../test_data.csv", index=False)  # save unscaled test set, this is what Streamlit app will use

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    if name in ["Logistic Regression", "kNN"]:  # these two need scaled input
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        probs = model.predict_proba(X_test_s)[:, 1]  # probability of class 1, needed for AUC
    else:  # tree/probability based models, raw features are fine
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "AUC": round(roc_auc_score(y_test, probs), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "F1": round(f1_score(y_test, preds), 4),
        "MCC": round(matthews_corrcoef(y_test, preds), 4)
    }
    with open(f"{name.replace(' ', '_')}.pkl", "wb") as f:  # save trained model for later use in app
        pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:  # save scaler too, app needs it for LR and kNN predictions
    pickle.dump(scaler, f)

with open("results.json", "w") as f:  # save all metrics, app reads this for the accuracy chart
    json.dump(results, f, indent=2)

print(pd.DataFrame(results).T)  # print final comparison table to console