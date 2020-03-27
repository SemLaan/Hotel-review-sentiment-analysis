from stop_words import get_stop_words
from Custom.Database import MysqlDB as Database
from Custom.text import TextCleaning
import pandas as pd
from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import plotly.express as px

# Collecting data
db = Database()
df = db.from_database('beef')
dfHead = df.head(1000)

# text cleaning
cleaner = TextCleaning()



# Tokenization
tokenizer = tokenize.RegexpTokenizer(r'\w+')
df['review'] = df['review'].apply(lambda x: tokenizer.tokenize(x))

# token cleaning


# untokenization
def untokenize(tokenized_review):
    return ' '.join(tokenized_review)

df['review'] = df['review'].apply(untokenize)


# Vectorization
stop_words = get_stop_words('english')
cv = CountVectorizer(stop_words=stop_words, max_features=1000)
cv = CountVectorizer(stop_words=stop_words)

vectorized_reviews = cv.fit_transform(df['review'])

cv.get_feature_names()



