import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

import os

train_data = pd.read_csv("/Users/narendrafuloria/Desktop/OPS/MLOPS_codes_for_different_Libraries/DVC/ML_PIPELINE_DEMO/data/processed/train_processed.csv")

test_data = pd.read_csv("/Users/narendrafuloria/Desktop/OPS/MLOPS_codes_for_different_Libraries/DVC/ML_PIPELINE_DEMO/data/processed/test_processed.csv")

X_train = train_data["content"].values
y_train = train_data["sentiment"].values

X_test = test_data["content"].values
y_test = test_data["sentiment"].values

## applying countvectoriser

vectoriser = CountVectorizer()

X_train_bow = vectoriser.fit_transform(X_train)

X_test_bow = vectoriser.transform(X_test)

train_df = pd.DataFrame(X_train_bow.toarray())
train_df["labels"]= y_train

test_df = pd.DataFrame(X_test_bow.toarray())

test_df["labels"] = y_test

data_path = os.path.join("data", "features")

os.makedirs(data_path, exist_ok= True)

train_df.to_csv(os.path.join(data_path, "train_fetures.csv"))
test_df.to_csv(os.path.join(data_path, "test_features.csv"))