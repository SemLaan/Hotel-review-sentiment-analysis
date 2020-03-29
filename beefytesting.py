from Custom.Database import MysqlDB as Database
import pandas as pd
from Custom import text
from stop_words import get_stop_words

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

db = Database()
df = db.from_database('model_tests')

stop = get_stop_words('english')

X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Cleaning
cleaner = text.TextCleaning()

#X = cleaner.lemmatize(X)
#X = cleaner.stem(X)
#X = cleaner.negate_corpus(X)
#X = cleaner.remove_numbers(X)


# multinomial Naive bayes
nb = MultinomialNB()
nb_pipe = make_pipeline(TfidfVectorizer(binary=False, ngram_range=(1, 2)), nb)

nb_results = cross_validate(nb_pipe, X, y, cv=5, n_jobs=-1, verbose=10)
print(nb_results['test_score'].mean())

# SVC radial basis function kernel
svc = SVC(random_state=0, kernel='linear', verbose=True)
svc_pipe = make_pipeline(TfidfVectorizer(binary=True, ngram_range=(1, 2)), svc)

svc_pipe.fit(X_train, y_train)
svc_pipe.score(X_test, y_test)

svc_results = cross_validate(svc_pipe, X, y, cv=5, n_jobs=-1, verbose=3)
print(svc_results['test_score'].mean())




