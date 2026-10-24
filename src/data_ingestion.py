import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split

import os

path ='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'

df = pd.read_csv(path)

df.drop(columns = ["tweet_id"], inplace= True)

final_df = df[df["sentiment"].isin(["happiness", "sadness"])].copy()

final_df["sentiment"] = final_df["sentiment"].replace({"happiness": 1, "sadness":0})


train_df, test_df = train_test_split(final_df, test_size= 0.2, random_state= 42)

data_path = os.path.join("data", "raw")

os.makedirs(data_path)

train_df.to_csv(os.path.join(data_path,"train.csv"))

test_df.to_csv(os.path.join(data_path, "test.csv"))



