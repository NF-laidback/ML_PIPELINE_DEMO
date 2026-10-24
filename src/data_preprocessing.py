import numpy as np
import pandas as pd
import re
import os
import nltk
import string
from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer, WordNetLemmatizer

# fetching data from raw folder

train_df = pd.read_csv("/Users/narendrafuloria/Desktop/OPS/MLOPS_codes_for_different_Libraries/DVC/ML_PIPELINE_DEMO/data/raw/train.csv")

test_df = pd.read_csv("/Users/narendrafuloria/Desktop/OPS/MLOPS_codes_for_different_Libraries/DVC/ML_PIPELINE_DEMO/data/raw/test.csv")

stopwords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def remove_stopwords(text):
    words = text.split()

    return " ".join(word for word in words if word not in stopwords)
from nltk import pos_tag
from nltk.corpus import wordnet

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

def lemmatize_text_pos(text):
    tagged = pos_tag(text.split())
    return ' '.join(lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged)

def text_processing(text):

    # removing urls and removing punctuations with space

    # 1. Remove URLs (http://, https://, and www. links)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 2. Replace every punctuation character with a space
    text = re.sub(r'[^\w\s]', ' ', text)
    # 3. Collapse repeated whitespace left behind
    text = re.sub(r'\s+', ' ', text).strip()

    # converting the text to lowercase

    text = text.lower()

    # removing stopwords

    remove_stopwords(text)

    # applying lemmatization on text

    text = lemmatize_text_pos(text)
    return text

train_df["content"] = train_df["content"].apply(text_processing)

test_df["content"] = test_df["content"].apply(text_processing)

path = os.path.join("data", "processed")

os.makedirs(path, exist_ok = True)

train_df.to_csv(os.path.join(path, "train_processed.csv"))
test_df.to_csv(os.path.join(path, "test_processed.csv"))