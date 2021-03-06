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
from joblib import load
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


class TrainedModel:

    def __init__(self, model):

        self.model = load('trained_models/' + model + '.joblib')
        self.model_type = model
        self.cleaner = text.TextCleaning()

    def predict(self, X):
        
        X = self.cleaner.to_lower(X)

        if self.model_type == 'svc' or self.model_type == 'naive_bayes':

            X = self.cleaner.remove_numbers(X)
            X = self.cleaner.lemmatize(X)
            X = self.cleaner.negate_corpus(X)

        else:

            X = self.cleaner.stem(X)

        return self.model.predict(X)

    def score(self, X, y):

        X = self.cleaner.to_lower(X)

        if self.model_type == 'random_forest':

            X = self.cleaner.stem(X)
            
        else:
            
            X = self.cleaner.remove_numbers(X)
            X = self.cleaner.lemmatize(X)
            X = self.cleaner.negate_corpus(X)

            

        return self.model.score(X, y)



# fetching data
db = Database()
df = db.from_database('test')

df = df.sample(10000, random_state=0)

X = df['review']
y = df['label']


# fetching models
nb = TrainedModel('naive_bayes')
svc = TrainedModel('svc')
rf = TrainedModel('random_forest')
svc_good_prune = TrainedModel('good_svc_pruned')
svc_bad_prune = TrainedModel('bad_svc_pruned')


# validating with test set
nb.score(X, y)
svc.score(X, y)
rf.score(X, y)
svc_good_prune.score(X, y)
svc_bad_prune.score(X, y)


# plotting confusion matrices
cleaner = text.TextCleaning()

X_a = cleaner.remove_numbers(X)
X_a = cleaner.lemmatize(X_a)
X_a = cleaner.negate_corpus(X_a)


X_b = cleaner.stem(X)

plot_confusion_matrix(nb.model, X_a, y, normalize='all', display_labels=['negatief', 'positief'])
plt.show()
plt.clf()
plot_confusion_matrix(svc.model, X_a, y, normalize='all', display_labels=['negatief', 'positief'])
plt.show()
plt.clf()
plot_confusion_matrix(rf.model, X_b, y, normalize='all', display_labels=['negatief', 'positief'])
plt.show()
plt.clf()



# testing my own reviews
my_reviews = pd.Series([
    'This hotel is bad the windows were small and the bread was old', 
    'The staff was rude and we got attacked by a chimp',
    'very nice hotel', 
    'comfy beds and respectfull staff',
    'Wendy s spot is ideal for access to the South Shore and makes drives to both Waimea Canyon and Hanalei manageable The house is great and gets a consistent breeze throughout  On top of this  Wendy is a super considerate and thoughtful host who responds quickly and cares about your stay'
])
my_reviews_labels = pd.Series([0, 0, 1, 1, 1])

nb.score(my_reviews, my_reviews_labels)
svc.score(my_reviews, my_reviews_labels)
rf.score(my_reviews, my_reviews_labels)
svc_good_prune.score(my_reviews, my_reviews_labels)
svc_bad_prune.score(my_reviews, my_reviews_labels)


