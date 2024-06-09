# -*- coding: utf-8 -*-
"""Vocab_Construction.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from google.colab import drive
drive.mount('/content/drive')
!ls "/"

"""#Load Data for split 1
Reviews from the train and test dataset along with their coresponding sentiment are combined into a data frame. This combined dataset is utilized for generating the vocabulary. By combining the reviews from the test and train it ensures the comprehensiveness of the vocabulary necessary for accurate predictions






"""

train = pd.read_csv("/split_1/train.tsv", sep='\t', header=0, dtype=str)
train['review'] = train['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

test = pd.read_csv("/split_1/test.tsv", sep='\t', header=0, dtype=str)
test['review'] = test['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

test_y = pd.read_csv("/split_1/test_y.tsv", sep='\t', header=0, dtype=str)

"""#Combine test & train reviews for split 1"""

#Combine test reviews with its corresponding sentiment
merged_df = pd.merge(test[['id', 'review']], test_y[['id', 'sentiment']], on='id', how='inner')
new_test_df = merged_df[['review', 'sentiment']]

#Combine test and train df
combined_df = pd.concat([train, new_test_df], axis=0)

#Remove the id column
if 'id' in combined_df.columns:
    combined_df.drop('id', axis=1, inplace=True)
print(combined_df)

"""#Declare stopwords"""

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

"""#Create inital vocabulary"""

vectorizer = TfidfVectorizer(
    stop_words=stop_words,          # Remove stop words
    lowercase=True,                 # Convert to lowercase
    ngram_range=(1, 4),             # Use 1- to 4-gram
    min_df=0.001,                   # Minimum term frequency
    max_df=0.5,                     # Maximum document frequency
    token_pattern=r"\b[\w+\|']+\b"  # Use word tokenizer to treat words with apostrophes as a single token
)
dtm_train = vectorizer.fit_transform(combined_df['review'])

"""#Trim vocabulary using LogisticRegression
Then to reduce the vocabulary size to 1000 words , we employ Logistic Regression with an L1 penalty, a liblinear solver, and a C value of 0.548. The parameter C represents inverse of regularization strength, where smaller values specify stronger regularization. After trial and error, the parameter value of .548 of C resulted in exactly 1000 nonzero coefficients. These indices correspond to the features (words or phrases) that the model has identified as important for predicting sentiment.
Therefore, the indices of 1000 nonzero coefficients are used to trim features from the vectorizer to obtain a final vocabulary of 1000 words.
"""

# Fit LogisticRegression with l1 penalty on the transformed training data dtm_train
logistic_reg_l1 = LogisticRegression(penalty='l1', solver='liblinear',max_iter=10000, C = .549)
logistic_reg_l1.fit(dtm_train, combined_df['sentiment'])

#Identify the indices where the coefficients obtained from the logistic regression model are non-zero
nonzero_indices = np.where(logistic_reg_l1.coef_ != 0)[1]
len(nonzero_indices)

#Obtain first 1000 non zero coefficients to trim vocabulary
nonzero_indices = np.where(logistic_reg_l1.coef_ != 0)[1][:1000]
len(nonzero_indices)

#Generate the final vocabulary to length 1000
final_vocab = np.array(vectorizer.get_feature_names_out())[nonzero_indices]
len(final_vocab)

"""Save final vocabulary to myvocab.txt"""

file_path = '/split_1/myvocab.txt'

# Write the vocabulary to a text file
with open(file_path, 'w') as file:
    for word in final_vocab:
        file.write(word + '\n')
