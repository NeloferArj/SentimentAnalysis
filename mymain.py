# -*- coding: utf-8 -*-
"""mymain.ipynbt
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


stop_words = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll",
    "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's",
    "her", "hers", "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re",
    "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn",
    "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn",
    "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren",
    "weren't", "won", "won't", "wouldn", "wouldn't"
]

# Load training data
# train_data_path = f'/split_{split}/train.tsv'
# train = pd.read_csv(train_data_path, sep='\t', header=0, dtype=str)
train = pd.read_csv("train.tsv", sep='\t', header=0, dtype=str)
train['review'] = train['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

# Load test data
# test_data_path = f'/split_{split}/test.tsv'
# test = pd.read_csv(test_data_path, sep='\t', header=0, dtype=str)
test = pd.read_csv("test.tsv", sep='\t', header=0, dtype=str)
test['review'] = test['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)


# Initialize an empty list to store vocabulary
top_vocab = []

# Load vocabulary from myvocab.txt
# file_path = '/split_1/myvocab.txt'
file_path = 'myvocab.txt'

# Read the contents of myvocab.txt
with open(file_path, 'r') as file:
    for word in file:
        top_vocab.append(word.strip())  # Append each word after stripping newline characters or whitespace

new_vectorizer = TfidfVectorizer(vocabulary=top_vocab,
    stop_words=stop_words,          # Remove stop words
    lowercase=True,                 # Convert to lowercase
    ngram_range=(1, 4),             # Use 1- to 4-gram
    min_df=0.001,                   # Minimum term frequency
    max_df=0.5,                     # Maximum document frequency
    token_pattern=r"\b[\w+\|']+\b"  # Use word tokenizer to treat words with apostrophes as a single token
)

# Fit TfidfVectorizer using trimmed vocabulary
dtm_train_1k = new_vectorizer.fit_transform(train['review'])
dtm_test_1k = new_vectorizer.fit_transform(test['review'])

#Generate Predictions
logistic_reg_l2 = LogisticRegression(penalty='l2', solver='liblinear',max_iter=10000)
logistic_reg_l2.fit(dtm_train_1k, train['sentiment'])
probabilities = logistic_reg_l2.predict_proba(dtm_test_1k)[:, 1]

#Export mysubmission.csv
result_df = test[['id']].copy()
result_df['probability'] = probabilities
result_df.to_csv('result.csv', columns=['id', 'probability'], header=['id', 'prob'], index=False)
