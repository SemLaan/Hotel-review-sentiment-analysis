# Collect Data
from Custom.Database import MysqlDB as Database
import pandas as pd
from numpy.random import RandomState
db = Database()
df = db.from_database('beef')
testingDF = df.sample(10000, random_state=0)
dfHead = df.head(1000)

X = testingDF['review']
y = testingDF['label']


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

# Making a test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)





tfidfBinary = TfidfVectorizer(stop_words='english', strip_accents='ascii', binary=True)
svc = SVC(random_state=0, cache_size=500)
pipeline = make_pipeline(tfidfBinary, svc)
pipeline.steps

param_grid = {
    'tfidfvectorizer__max_features': [1000, 3000, 5000, 6000],
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
}

grid = GridSearchCV(pipeline, param_grid, n_jobs=3, cv=5, )
grid.fit(X, y)
grid.best_score_
grid.best_params_



beef = grid.best_estimator_

beef.fit(X_train, y_train)
beef.score(X_test, y_test)




# loading model
aap = load('100000.joblib')
aap.score(X_test.iloc[:1000], y_test.iloc[:1000])



# saving model
dump(pipeline, '100000.joblib')


