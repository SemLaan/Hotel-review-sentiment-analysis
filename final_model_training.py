from Custom.Database import MysqlDB as Database
import pandas as pd
from Custom import text
from stop_words import get_stop_words

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from joblib import dump


# fetching data
db = Database()
df = db.from_database('train')
X = df['review']
y = df['label']


# Initializing cleaning class
cleaner = text.TextCleaning()


# Text cleaning
X_svm_nb = cleaner.remove_numbers(X)
X_svm_nb = cleaner.lemmatize(X_svm_nb)
X_svm_nb = cleaner.negate_corpus(X_svm_nb)

X_rf = cleaner.stem(X)




# SVM
svc = SVC(random_state=0, cache_size=1000, C=1.0, kernel='rbf', gamma=1)
svc_vectorizer = TfidfVectorizer(binary=True)
svc_pipe = make_pipeline(svc_vectorizer, svc)
svc_pipe.fit(X, y)

dump(svc_pipe, 'trained_models/svc.joblib')


# Random Forest
rf = RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=200, criterion='gini', max_depth=None)
rf_vectorizer = CountVectorizer()
rf_pipe = make_pipeline(rf_vectorizer, rf)
rf_pipe.fit(X, y)

dump(rf_pipe, 'trained_models/random_forest.joblib')


# Naive Bayes
nb = MultinomialNB()
nb_vectorizer = CountVectorizer(ngram_range=(1, 2))
nb_pipe = make_pipeline(nb_vectorizer, nb)
nb_pipe.fit(X, y)

dump(nb_pipe, 'trained_models/naive_bayes.joblib')


