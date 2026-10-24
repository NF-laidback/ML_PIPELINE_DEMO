import numpy as np

import pandas as pd

import xgboost as xgb
import pickle
import os

train_df = pd.read_csv("/Users/narendrafuloria/Desktop/OPS/MLOPS_codes_for_different_Libraries/DVC/ML_PIPELINE_DEMO/data/features/train_fetures.csv")

X_train = train_df.drop(columns= ["labels"])

y_train = train_df["labels"]

model = xgb.XGBClassifier(use_label_encoder = False, eval_metric = "mlogloss")

model.fit(X_train, y_train)

#save

os.makedirs("model", exist_ok= True)

model_path = os.path.join("model", "trained_model.pkl") 

with open(model_path, "wb") as file:
    pickle.dump(model,file)



