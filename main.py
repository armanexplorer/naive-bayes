from os import name
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import pickle


# LABELS = ['اجتماعی', 'اديان', 'اقتصادی', 'سیاسی', 'فناوري', 'مسائل راهبردي ايران', 'ورزشی']
with open("persian-stopwords/persian", encoding="utf-8") as f:
    STOP_WORDS = f.read().splitlines()
# STOP_WORDS.append(['amp', 'nbsp', ])
# print(STOP_WORDS)
# Load all documents
docs = load_files("All")
train_docs = load_files("Train")
test_docs = load_files("Test")
X, y = docs.data, docs.target
X_train, y_train = train_docs.data, train_docs.target
X_test, y_test = test_docs.data, test_docs.target

# Vectorize data
vec = CountVectorizer(max_features=500, stop_words=STOP_WORDS)
X = vec.fit_transform(X).toarray()

train_vec = CountVectorizer(vocabulary=vec.vocabulary_.keys())
test_vec = CountVectorizer(vocabulary=vec.vocabulary_.keys())
X_train = train_vec.fit_transform(X_train).toarray()
X_test = test_vec.fit_transform(X_test).toarray()
print(X_train)
print(X_test)

# Create dateframes besed on frequent words
# df_test = pd.DataFrame(0, index=range(len(X_test)), columns=vec.get_feature_names())
# df_train = pd.DataFrame(0, index=range(len(X_train)), columns=vec.get_feature_names())
# docs_filenames = [x.split('\\')[2] for x in docs.filenames]
# for test_index, test_filename in enumerate(test_docs.filenames):
#     df_test.values[test_index] = X[docs_filenames.index(test_filename.split('\\')[2])]
# for train_index, train_filename in enumerate(train_docs.filenames):
#     df_train.values[train_index] = X[docs_filenames.index(train_filename.split('\\')[2])]

# Create Model

# print(list(y_train))
# print(df_train)
# print(list(df_test.values))

clf = MultinomialNB()
# clf.fit(df_train, y_train)
clf.fit(X_train, y_train)

# Evaluate Model
# y_pred = clf.predict(df_test)
y_pred = clf.predict(X_test)
# print(y_test)
# print(y_pred)
print(accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

# print([df_test.columns[index] for index, el in enumerate((df_test==0).all()) if el == True])


# Write Model to files
# with open('text_classifier', 'wb') as picklefile:
#     pickle.dump(clf,picklefile)



# print(term_freq_per_doc_table)


# docs.filenames.where(arr='All\\اقتصادی\\13890626-txt-1614537_utf.txt')


# Concat dataframes
# term_freq_per_doc_table.update(train_term_freq_per_doc_table)
# print(term_freq_per_doc_table)

# term_table = pd.DataFrame(X, columns=vec.get_feature_names())
# print(term_table.sum(axis=0).sort_values())

# TFIDF
# tfidfconverter = TfidfTransformer()
# X = tfidfconverter.fit_transform(X).toarray()

# TFIDFVectorize
# tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# X = tfidfconverter.fit_transform(documents).toarray()

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train)
# print(y_train)

# classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
# classifier.fit(X_train, y_train) 
# y_pred = classifier.predict(X_test)




# stmt_docs = [row['sent'] for index,row in training_data.iterrows() if row['class'] == 'stmt']
# vec_s = CountVectorizer()
# X_s = vec_s.fit_transform(stmt_docs)
# tdm_s = pd.DataFrame(X_s.toarray(), columns=vectorizer.get_feature_names())
# tdm_s




# vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# X = vectorizer.fit_transform(documents).toarray()