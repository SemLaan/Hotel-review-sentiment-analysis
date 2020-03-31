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
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from joblib import dump



# fetching data
db = Database()
df = db.from_database('test')
X = df['review']
y = df['label']



# Initializing cleaning class
cleaner = text.TextCleaning()


# Text cleaning
X_svm_nb = cleaner.remove_numbers(X)
X_svm_nb = cleaner.lemmatize(X_svm_nb)
X_svm_nb = cleaner.negate_corpus(X_svm_nb)

X_rf = cleaner.stem(X)



X_train, X_test, y_train, y_test = train_test_split(X_svm_nb, y, test_size=0.10)


vectorizer = TfidfVectorizer(binary=True, ngram_range=(1, 2))

X_train_rf = vectorizer.fit_transform(X_train)

rf = RandomForestClassifier(n_estimators=400, verbose=100, n_jobs=-1, random_state=0, max_samples=5000)

rf.fit(X_train_rf, y_train)

feature_selector = SelectFromModel(rf, prefit=True, max_features=100000)


svc_set = pd.concat([X_train, y_train], axis=1)
svc_set = svc_set.sample(100000, random_state=0)

svc_X = svc_set['review']
svc_y = svc_set['label']

svc_X = vectorizer.transform(svc_X)
svc_X = feature_selector.transform(svc_X)


svc = SVC(cache_size=1000, random_state=0)

svc.fit(svc_X, svc_y)


final_pipe = make_pipeline(vectorizer, feature_selector, svc)

# 95.92% accuracy
final_pipe.score(X_test, y_test)


dump(final_pipe, 'trained_models/good_svc_pruned')

