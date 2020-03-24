# Collect Data
from Custom.Database import MysqlDB as Database
import pandas as pd
db = Database()
df = db.from_database('beef')
testingDF = df.sample(15000, random_state=0)
dfHead = df.head(1000)

X = testingDF['review']
y = testingDF['label']

# Calculating score when guessing either all positive or all negative
# Score: 0.5
posandneg = df.groupby('label').count()
print(max(posandneg.iloc[0, 0], posandneg.iloc[1, 0]) / posandneg['review'].sum())


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

# Making a test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

cv = CountVectorizer(stop_words='english', strip_accents='ascii', max_features=1000)
cvlr = LogisticRegression()
tfidf = TfidfVectorizer(stop_words='english', strip_accents='ascii', max_features=1000)
tfidflr = LogisticRegression()

cv_pipeline = make_pipeline(cv, cvlr)
tfidf_pipeline = make_pipeline(tfidf, tfidflr)

# Trying logistic regression with Count Vectorizer
# Score: 0.90
cv_pipeline.fit(X_train, y_train)
cv_pipeline.score(X_test, y_test)

# Trying logistic regression with TFIDF Vectorizer
# Score: 0.91
tfidf_pipeline.fit(X_train, y_train)
tfidf_pipeline.score(X_test, y_test)


# !!!!!!!!! Only TFIDF Vectorizer from here on out

# Trying random forest classifier with 100 trees
# Score: 0.90
rf = RandomForestClassifier()
rf_pipeline = make_pipeline(tfidf, rf)
rf_pipeline.fit(X_train, y_train)
rf_pipeline.score(X_test, y_test)


# Trying ada boost classifier with 50 trees
# Score: 0.87
ada = AdaBoostClassifier()
ada_pipeline = make_pipeline(tfidf, ada)
ada_pipeline.fit(X_train, y_train)
ada_pipeline.score(X_test, y_test)


# Trying gradient boost classifier with 100 trees
# Score: 0.87
gradient = GradientBoostingClassifier()
gradient_pipeline = make_pipeline(tfidf, gradient)
gradient_pipeline.fit(X_train, y_train)
gradient_pipeline.score(X_test, y_test)


# Trying support vector machine with polynomial kernel
# Score: 0.89
svc = SVC(degree=3, kernel='poly')
svc_pipeline = make_pipeline(tfidf, svc)
svc_pipeline.fit(X_train, y_train)
svc_pipeline.score(X_test, y_test)


# Trying support vector machine with brf kernel
# Score: 0.92
brf = SVC(probability=True)
brf_pipeline = make_pipeline(tfidf, brf)
brf_pipeline.fit(X_train, y_train)
brf_pipeline.score(X_test, y_test)


# Trying naive bayes
# Score: 0.90
nb = MultinomialNB()
nb_pipeline = make_pipeline(tfidf, nb)
nb_pipeline.fit(X_train, y_train)
nb_pipeline.score(X_test, y_test)


# Grid search on SVM with brf kernel (5-fold cross validation)
# Best score: 0.91 (with cross fold)  0.92 (on full data set)
from sklearn.model_selection import GridSearchCV

tfX_train = tfidf.fit_transform(X_train)
tfX_test = tfidf.transform(X_test)

parameter_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(SVC(), parameter_grid, n_jobs=4)
grid.fit(tfX_train, y_train)

grid.best_params_
grid.best_score_


brf = grid.best_estimator_
brf_pipeline = make_pipeline(tfidf, brf)
brf_pipeline.fit(X_train, y_train)
brf_pipeline.score(X_test, y_test)



# Neural network attempts
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

scaler = StandardScaler(with_mean=False)
sX_train = scaler.fit_transform(tfX_train)
sX_test = scaler.transform(tfX_test)

nn = MLPClassifier(hidden_layer_sizes=(80, 20, 5,))
nn.fit(sX_train, y_train)
nn.score(sX_test, y_test)







# stacking

prob_lr = tfidf_pipeline.predict_proba(X_train)
prob_rf = rf_pipeline.predict_proba(X_train)
prob_brf = brf_pipeline.predict_proba(X_train)
prob_grad = gradient_pipeline.predict_proba(X_train)
layer_2_train = pd.DataFrame({'lr': prob_lr[:,0], 'rf': prob_rf[:,0], 'brf': prob_brf[:,0], 'grad': prob_grad[:,0]})
layer_2_train.info()
prob_lr = tfidf_pipeline.predict_proba(X_test)
prob_rf = rf_pipeline.predict_proba(X_test)
prob_brf = brf_pipeline.predict_proba(X_test)
prob_grad = gradient_pipeline.predict_proba(X_test)
layer_2_test = pd.DataFrame({'lr': prob_lr[:,0], 'rf': prob_rf[:,0], 'brf': prob_brf[:,0], 'grad': prob_grad[:,0]})


stackJudge = SVC(random_state=0, C=1)
stackJudge.fit(layer_2_train, y_train)
stackJudge.score(layer_2_test, y_test)


# binary vs scaling vs normal tfidf
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=False)

tfidf = TfidfVectorizer(stop_words='english', strip_accents='ascii', max_features=1000)
tfidfBinary = TfidfVectorizer(stop_words='english', strip_accents='ascii', max_features=1000, binary=True)

svc = SVC()
svc_binary = SVC()

svc_pipeline = make_pipeline(tfidf, svc)
binary_pipeline = make_pipeline(tfidfBinary, svc_binary)


svc_pipeline.fit(X_train, y_train)
svc_pipeline.score(X_test, y_test)


binary_pipeline.fit(X_train, y_train)
binary_pipeline.score(X_test, y_test)


scaler_pipeline = make_pipeline(tfidf, scaler, SVC())

scaler_pipeline.fit(X_train, y_train)
scaler_pipeline.score(X_test, y_test)


# final test i guess?
tfidfBinary = TfidfVectorizer(stop_words='english', strip_accents='ascii', max_features=2500, binary=True)
svc_binary = SVC()
binary_pipeline = make_pipeline(tfidfBinary, svc_binary)

binary_pipeline.fit(X_train, y_train)
binary_pipeline.score(X_test, y_test)





#Cmaj7 Fmaj7^-1 G^-1
