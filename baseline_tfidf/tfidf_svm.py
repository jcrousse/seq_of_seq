from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def tfidf_svm(texts, y):
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('prediction', LinearSVC())])
    model.fit(texts, y)
    return model
