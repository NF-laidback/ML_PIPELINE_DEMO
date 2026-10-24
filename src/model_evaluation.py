import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
import os
import json

model = pickle.load(open(os.path.join("model", "trained_model.pkl"), "rb"))

test_df = pd.read_csv(os.path.join("data", "features", "test_features.csv"))

X_test = test_df.drop(columns= ["labels"])

y_test = test_df["labels"]

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test,y_pred)

recall = recall_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict = {
    "accuracy":accuracy,
    "precision":precision,
    "recall": recall,
    "roc_auc" : roc_auc
}

os.makedirs("metrics", exist_ok= True)

metrics_path = os.path.join("metrics", "metrics.json")

with open(metrics_path ,"w") as file:
    json.dump(metrics_dict, file, indent= 4)
