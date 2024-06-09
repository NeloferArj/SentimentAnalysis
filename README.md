# SentimentAnalysis
Section 1: Technical Details:
<br />
Generating Vocabulary:
	We start off by loading the provided datasets, including train.tsv, test.tsv, and test_y.tsv. We then merge the test and test_y data frames based on their id columns to create a merged data frame with columns id, review, and sentiment. This data frame is concatenated with the train data frame to create a combined data frame with the reviews and sentiments from the test and train datasets. This combined data frame will be used to generate our vocabulary. To initialize the vocabulary, a stop words lists is decaled for subsequent use in the TfidfVectorizer. Utilizing this vectorizer with the specified parameters such as (such as ngram range of 1-4, min_df = .001, max_df = .5, and token pattern = r”\b[\w+\|']+\b"), the reviews from combined data frame are transformed into a document-term matrix (dtm_train). Then to reduce the vocabulary size to 1000 words , we employ Logistic Regression with an L1 penalty, a liblinear solver, and a C value of 0.548. The parameter C represents inverse of regularization strength, where smaller values specify stronger regularization. After trial and error, the parameter value of .548 of C resulted in exactly 1000 non-zero coefficients. These indices correspond to the features (words or phrases) that the model has identified as important for predicting sentiment. Therefore, these indices were used to trim features from the vectorizer to obtain a final vocabulary of 1000 words. This vocab is then exported as a txt file “myvocab.txt”, this final vocabulary will be used for testing all 5 splits.
<br />
<br />
Generating Predictions:
	For generating predictions, the vocabulary from “myvocab.txt” is loaded and used in a new TfidfVectorizer along with the previously set parameters. This vectorizer is applied to both train and test datasets, generating dtm_train_1k and dtm_test_1k. The logistic regression classifier, utilizing an L2 penalty, liblinear solver, and max_iter 10000, is trained on dtm_train_1k and the corresponding sentiment from the train dataset. Predictions are then generated using the predict_proba function on previously generated dtm_test_1k. The performance evaluation is measured using the Area Under the Curve (AUC) via roc_auc_score, comparing the predicted probabilities of the sentiment being positive against the sentiment column from test_y.
<br />
<br />
Section 2: Performance Metrics: 
<br />
Performance Metric: The metric used to evaluate the models was the Area Under the Curve (AUC). 
<br />
Vocabulary size: 1,000 words
<br />
Performance accuracy of the model on test data:
<br />
Fold	AUC	Execution Time (Seconds)
<br />
1	0.96309	16.76 secs
<br />
2	0.96171	16.36 secs
<br />
3	0.96179	16.07 secs
<br />
4	0.96291	18.77 secs
<br />
5	0.96197	15.86 secs
<br />
<br />
Computer System: Windows, 2.11 GHz, 8GB memory for all 5/training test splits 
<br />
Dataset: https://paperswithcode.com/dataset/imdb-movie-reviews
