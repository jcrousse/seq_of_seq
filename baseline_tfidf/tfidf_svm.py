import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC


def tfidf_svm(texts, y):
    model = Pipeline([('tfidf', TfidfVectorizer()), ('prediction', LinearSVC())])
    return model.fit(texts, y)
