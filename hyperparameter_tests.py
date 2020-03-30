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


# fetching data
db = Database()
df = db.from_database('model_tests')
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
svc = SVC(random_state=0, cache_size=1000)
svc_vectorizer = TfidfVectorizer(binary=True)
svc_pipe = make_pipeline(svc_vectorizer, svc)

grid = [
    {
        'svc__kernel': ['linear'],
        'svc__C': [0.1, 1, 10]
    },
    {
        'svc__kernel': ['poly'],
        'svc__C': [0.1, 1, 10],
        'svc__degree': [2, 3]
    },
    {
        'svc__kernel': ['rbf'],
        'svc__C': [0.1, 1, 10],
        'svc__gamma': [0.1, 1, 10]
    }
]

svc_grid = GridSearchCV(svc_pipe, grid, n_jobs=-1, verbose=10)
svc_results = svc_grid.fit(X_svm_nb, y)

print(svc_results.best_params_)
print(svc_results.best_score_)

# Random Forest
rf = RandomForestClassifier(random_state=0, n_jobs=-1)
rf_vectorizer = CountVectorizer()
rf_pipe = make_pipeline(rf_vectorizer, rf)
rf_pipe.steps
rf_grid = {
    'randomforestclassifier__n_estimators': [75, 100, 200],
    'randomforestclassifier__criterion': ['gini', 'entropy'],
    'randomforestclassifier__max_depth': [5, 10, None]
}

rf_grid_search = GridSearchCV(rf_pipe, rf_grid, n_jobs=-1, verbose=10)
rf_results = rf_grid_search.fit(X_rf, y)

print(rf_results.best_params_)
print(rf_results.best_score_)

# Naive Bayes
nb = MultinomialNB()
nb_vectorizer = CountVectorizer(ngram_range=(1, 2))
nb_pipe = make_pipeline(nb_vectorizer, nb)

nb_grid = {}

nb_grid_search = GridSearchCV(nb_pipe, nb_grid, n_jobs=-1, verbose=10)
nb_results = nb_grid_search.fit(X_svm_nb, y)

print(nb_results.best_estimator_)
print(nb_results.best_score_)









