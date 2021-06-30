import pickle

import pandas as pd
import seaborn as sns
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

sns.set_theme()

import warnings

import matplotlib.pyplot as plt

# Prevent from stop-words warning
warnings.filterwarnings("ignore")

# Load persian stop-words
with open("persian-stopwords/persian", encoding="utf-8") as f:
    STOP_WORDS = f.read().splitlines()
EXC_LIST = ['amp', 'nbsp', '10', '15', '20']
STOP_WORDS.extend(EXC_LIST)
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

train_vec = CountVectorizer(
    vocabulary=vec.get_feature_names(), stop_words=STOP_WORDS)
test_vec = CountVectorizer(
    vocabulary=vec.get_feature_names(), stop_words=STOP_WORDS)
X_train = train_vec.fit_transform(X_train).toarray()
X_test = test_vec.fit_transform(X_test).toarray()

# Create Models
nb_model = MultinomialNB()

knn1_model = KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='auto')
knn5_model = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
knn15_model = KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='auto')
knn1_tfidf_model = KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='auto')
knn5_tfidf_model = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
knn15_tfidf_model = KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='auto')

# Train Model
nb_model.fit(X_train, y_train)
knn1_model.fit(X_train, y_train)
knn5_model.fit(X_train, y_train)
knn15_model.fit(X_train, y_train)

# Predict
nb_y_pred = nb_model.predict(X_test)
knn1_y_pred = knn1_model.predict(X_test)
knn5_y_pred = knn5_model.predict(X_test)
knn15_y_pred = knn15_model.predict(X_test)

# Evaluate Models
nb_accuracy = accuracy_score(y_test, nb_y_pred)
knn1_accuracy = accuracy_score(y_test, knn1_y_pred)
knn5_accuracy = accuracy_score(y_test, knn5_y_pred)
knn15_accuracy = accuracy_score(y_test, knn15_y_pred)

# TFIDF on KNN
tfidfconverter = TfidfTransformer()
X_train = tfidfconverter.fit_transform(X_train).toarray()
X_test = tfidfconverter.fit_transform(X_test).toarray()
# Train
knn1_tfidf_model.fit(X_train, y_train)
knn5_tfidf_model.fit(X_train, y_train)
knn15_tfidf_model.fit(X_train, y_train)
# Predict
knn1_tfidf_y_pred = knn1_tfidf_model.predict(X_test)
knn5_tfidf_y_pred = knn5_tfidf_model.predict(X_test)
knn15_tfidf_y_pred = knn15_tfidf_model.predict(X_test)
# Evaluate
knn1_tfidf_accuracy = accuracy_score(y_test, knn1_tfidf_y_pred)
knn5_tfidf_accuracy = accuracy_score(y_test, knn5_tfidf_y_pred)
knn15_tfidf_accuracy = accuracy_score(y_test, knn15_tfidf_y_pred)

# Create result DataFrames
nb_res = pd.DataFrame([[nb_accuracy]], columns=['Naive Bayes Model'])
knn_res = pd.DataFrame([[knn1_accuracy, knn5_accuracy, knn15_accuracy]], columns=['KNN1 Model', 'KNN5 Model', 'KNN15 Model'])
knn_tfidf_res = pd.DataFrame([[knn1_tfidf_accuracy, knn5_tfidf_accuracy, knn15_tfidf_accuracy]], columns=['KNN1_TFIDF Model', 'KNN5_TFIDF Model', 'KNN15_TFIDF Model'])

# Print accuracies
print("\n", nb_res.to_string(index=False))
print("\n", knn_res.to_string(index=False))
print("\n", knn_tfidf_res.to_string(index=False))

# Calculate confusion matrix
nb_cm = confusion_matrix(y_test, nb_y_pred)
knn1_cm = confusion_matrix(y_test, knn1_y_pred)
knn5_cm = confusion_matrix(y_test, knn5_y_pred)
knn15_cm = confusion_matrix(y_test, knn15_y_pred)
knn1_tfidf_cm = confusion_matrix(y_test, knn1_tfidf_y_pred)
knn5_tfidf_cm = confusion_matrix(y_test, knn5_tfidf_y_pred)
knn15_tfidf_cm = confusion_matrix(y_test, knn15_tfidf_y_pred)

# Print confusion matrix for knn5 TFIDF confusion matrix
print(knn5_tfidf_cm)
# cm_matrix = pd.DataFrame(data=cm, index=[], columns=[])
# print(accuracy)
# print(cm)

# Print measures like F-measure, precision and recall for Naive Bayes
print(classification_report(y_test, nb_y_pred))
# print(X.shape)

# Visualize confusion matrix
# heat_map = sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
# heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
# plt.xlabel('True Class', labelpad=10)
# plt.ylabel('Predicted Class', labelpad=10)
# plt.title('Confusion Matrix', y=1.07)
# plt.show()

# Log
# print(pd.DataFrame(X_train, columns=vec.get_feature_names()))
# print(pd.DataFrame(X_test, columns=vec.get_feature_names()))
# print(test_docs.filenames)

# Write Model to file
clf = nb_model
with open('naive_bayes_text_classifier', 'wb') as picklefile:
    pickle.dump(clf,picklefile)

#################################### END OF THE FIRST PART ####################################

# TFIDF
# tfidfconverter = TfidfTransformer()
# X = tfidfconverter.fit_transform(X).toarray()

# TFIDFVectorize
# tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# X = tfidfconverter.fit_transform(documents).toarray()


# vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))




# from sklearn.pipeline import Pipeline
# text_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', MultinomialNB()),
# ])



# from sklearn.model_selection import GridSearchCV
# parameters = {
#     'vect__ngram_range': [(1, 1), (1, 2)],
#     'tfidf__use_idf': (True, False),
#     'clf__alpha': (1e-2, 1e-3),
# }


# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
# gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])