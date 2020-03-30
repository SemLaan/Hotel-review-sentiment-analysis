from Custom.Database import MysqlDB as Database
import pandas as pd
from Custom import text
from stop_words import get_stop_words

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_validate

# fetching data
db = Database()
df = db.from_database('model_tests')
X = df['review']
y = df['label']

# initializing models
svc = SVC(cache_size=500, random_state=0)
rf = RandomForestClassifier(random_state=0)
nb = MultinomialNB()

# Initializing cleaning class and fetching stop words
cleaner = text.TextCleaning()
stop = get_stop_words('english')
cleaner.count_negations(X)

svc_pipe = make_pipeline(CountVectorizer(), svc)
rf_pipe = make_pipeline(CountVectorizer(), rf)
nb_pipe = make_pipeline(CountVectorizer(), nb)


# tests

# no cleaning / ngram 1-1
print(cross_validate(svc_pipe, X, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(rf_pipe, X, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(nb_pipe, X, y, n_jobs=-1)['test_score'].mean())

# Remove numbers
X_no_numbers = cleaner.remove_numbers(X)

print(cross_validate(svc_pipe, X_no_numbers, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(rf_pipe, X_no_numbers, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(nb_pipe, X_no_numbers, y, n_jobs=-1)['test_score'].mean())

# Stemming
X_stem = cleaner.stem(X)

print(cross_validate(svc_pipe, X_stem, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(rf_pipe, X_stem, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(nb_pipe, X_stem, y, n_jobs=-1)['test_score'].mean())

# Lemmatization
X_lemmatize = cleaner.lemmatize(X)

print(cross_validate(svc_pipe, X_lemmatize, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(rf_pipe, X_lemmatize, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(nb_pipe, X_lemmatize, y, n_jobs=-1)['test_score'].mean())

# stop words

svc_sw_pipe = make_pipeline(CountVectorizer(stop_words=stop), svc)
rf_sw_pipe = make_pipeline(CountVectorizer(stop_words=stop), rf)
nb_sw_pipe = make_pipeline(CountVectorizer(stop_words=stop), nb)

print(cross_validate(svc_sw_pipe, X, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(rf_sw_pipe, X, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(nb_sw_pipe, X, y, n_jobs=-1)['test_score'].mean())

# ngram 1-2
svc_ngram = make_pipeline(CountVectorizer(ngram_range=(1, 2)), svc)
rf_ngram = make_pipeline(CountVectorizer(ngram_range=(1, 2)), rf)
nb_ngram = make_pipeline(CountVectorizer(ngram_range=(1, 2)), nb)

print(cross_validate(svc_ngram, X, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(rf_ngram, X, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(nb_ngram, X, y, n_jobs=-1)['test_score'].mean())

# ngram 1-1 negation
X_negated = cleaner.negate_corpus(X)

print(cross_validate(svc_pipe, X_negated, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(rf_pipe, X_negated, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(nb_pipe, X_negated, y, n_jobs=-1)['test_score'].mean())

# ngram 1-2 negation
print(cross_validate(svc_ngram, X_negated, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(rf_ngram, X_negated, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(nb_ngram, X_negated, y, n_jobs=-1)['test_score'].mean())


# Vectorizer tests

# Count vectorizer
print(cross_validate(svc_pipe, X, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(rf_pipe, X, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(nb_pipe, X, y, n_jobs=-1)['test_score'].mean())

# TFIDF vectorizer
svc_tfidf = make_pipeline(TfidfVectorizer(), svc)
rf_tfidf = make_pipeline(TfidfVectorizer(), rf)
nb_tfidf = make_pipeline(TfidfVectorizer(), nb)

print(cross_validate(svc_tfidf, X, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(rf_tfidf, X, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(nb_tfidf, X, y, n_jobs=-1)['test_score'].mean())

# TFIDF binary vectorizer
svc_binary = make_pipeline(TfidfVectorizer(binary=True), svc)
rf_binary = make_pipeline(TfidfVectorizer(binary=True), rf)
nb_binary = make_pipeline(TfidfVectorizer(binary=True), nb)

print(cross_validate(svc_binary, X, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(rf_binary, X, y, n_jobs=-1)['test_score'].mean())
print(cross_validate(nb_binary, X, y, n_jobs=-1)['test_score'].mean())



