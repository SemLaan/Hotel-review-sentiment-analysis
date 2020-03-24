
import pandas as pd
from nltk import tokenize as tokenizers
from nltk.stem import PorterStemmer, WordNetLemmatizer

class TextCleaning:

    def __init__(self):

        return

    def remove_hyperlinks(self, corpus):
        
        corpus = corpus.str.replace(r"https?://t.co/[A-Za-z0-9]+", "https")
        return corpus


    def remove_numbers(self, corpus):
        
        corpus = corpus.str.replace(r"\w*\d\w*", "")
        return corpus

    def tokenize(self, corpus):
        
        tokenizer = tokenizers.RegexpTokenizer(r'\w+')
        corpus = corpus.apply(lambda x: tokenizer.tokenize(x))
        return corpus


    def untokenize(self, corpus):
        
        corpus = corpus.apply(
            lambda tokenized_review: ' '.join(tokenized_review)
        )
        return corpus

    def lemmatize(self, corpus):
        
        corpus = self.tokenize(corpus)

        lemmatizer = WordNetLemmatizer()
        corpus = corpus.apply(
            lambda tokens: [lemmatizer.lemmatize(token) for token in tokens]
        )

        return self.untokenize(corpus)

    def stem(self, corpus):

        corpus = self.tokenize(corpus)

        stemmer = PorterStemmer()
        corpus = corpus.apply(
            lambda tokens: [stemmer.stem(token) for token in tokens]
        )

        return self.untokenize(corpus)

    def to_lower(self, corpus):

        return corpus.apply(str.lower)

    def negate_corpus(self, corpus):

        corpus = corpus.apply(self.negate_sentence)
        return corpus

    def negate_sentence(self, sentence):

        sentence = sentence.lower()

        for word in appos:
            if word in sentence:
                sentence = sentence.replace(word, appos[word])

        return sentence.lower()






appos = {
    "aren't" : "are not",
    "can't" : "cannot",
    "couldn't" : "could not",
    "didn't" : "did not",
    "doesn't" : "does not",
    "don't" : "do not",
    "hadn't" : "had not",
    "hasn't" : "has not",
    "haven't" : "have not",
    "he'd" : "he would",
    "he'll" : "he will",
    "he's" : "he is",
    "i'd" : "I would",
    "i'll" : "I will",
    "i'm" : "I am",
    "isn't" : "is not",
    "it's" : "it is",
    "it'll":"it will",
    "i've" : "I have",
    "let's" : "let us",
    "mightn't" : "might not",
    "mustn't" : "must not",
    "shan't" : "shall not",
    "she'd" : "she would",
    "she'll" : "she will",
    "she's" : "she is",
    "shouldn't" : "should not",
    "that's" : "that is",
    "there's" : "there is",
    "they'd" : "they would",
    "they'll" : "they will",
    "they're" : "they are",
    "they've" : "they have",
    "we'd" : "we would",
    "we're" : "we are",
    "weren't" : "were not",
    "we've" : "we have",
    "what'll" : "what will",
    "what're" : "what are",
    "what's" : "what is",
    "what've" : "what have",
    "where's" : "where is",
    "who'd" : "who would",
    "who'll" : "who will",
    "who're" : "who are",
    "who's" : "who is",
    "who've" : "who have",
    "won't" : "will not",
    "wouldn't" : "would not",
    "you'd" : "you would",
    "you'll" : "you will",
    "you're" : "you are",
    "you've" : "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll":" will",
}
