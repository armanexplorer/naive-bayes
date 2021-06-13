from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load persian stop-words
with open("persian-stopwords/persian", encoding="utf-8") as f:
    STOP_WORDS = f.read().splitlines()
# EXC_LIST = ['amp', 'nbsp', '10', '15', '20']
# STOP_WORDS.extend(EXC_LIST)
# STOP_WORDS = []

# Load documents
docs = load_files("All")
train_docs = load_files("Train")
test_docs = load_files("Test")
X, y = docs.data, docs.target
X_train, y_train = train_docs.data, train_docs.target
X_test, y_test = test_docs.data, test_docs.target

# Vectorize data
vec = CountVectorizer(max_features=500, stop_words=STOP_WORDS)
X = vec.fit_transform(X).toarray()

train_vec = CountVectorizer(vocabulary=vec.get_feature_names(), stop_words=STOP_WORDS)
test_vec = CountVectorizer(vocabulary=vec.get_feature_names(), stop_words=STOP_WORDS)
X_train = train_vec.fit_transform(X_train).toarray()
X_test = test_vec.fit_transform(X_test).toarray()

# Create Model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate Model
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

# Log
# print(pd.DataFrame(X_train, columns=vec.get_feature_names()))
# print(pd.DataFrame(X_test, columns=vec.get_feature_names()))
# print(len(vec.get_feature_names()))
# print(len(vec.vocabulary_))

# Write Model to file
with open('text_classifier', 'wb') as picklefile:
    pickle.dump(clf,picklefile)

#################################### END OF THE FIRST PART ####################################

# TFIDF
# tfidfconverter = TfidfTransformer()
# X = tfidfconverter.fit_transform(X).toarray()

# TFIDFVectorize
# tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# X = tfidfconverter.fit_transform(documents).toarray()


# vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))