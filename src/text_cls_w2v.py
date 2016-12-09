import os
import logging
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from tabulate import tabulate
from gensim.models import  Word2Vec
import gensim

from data_loader import load_data, load_w2v_data

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    filename='./result/w2v/text_w2vcls.log',
                    filemode='wb+')

data_root = os.getcwd()[:-3] + "data/"
"""
# Using the pre-trained models from GloVe(http://nlp.stanford.edu/projects/glove/)
# GloVe is an unsupervised learning algorithm for obtaining vector representations for words.
"""
# 50d -> word vector length is 50
GLOVE_6B_50D_PATH = data_root + "glove.6B.50d.txt"
# 300d -> word vector length is 300
GLOVE_840B_300D_PATH = data_root + "glove.840B.300d.txt"

with open(GLOVE_6B_50D_PATH, "rb") as lines:
    word2vec = {
        line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines
        }

filename = data_root + "r8-no-stop.txt"
X, Y = load_data(filename)

glove_small, glove_big = load_w2v_data(X, GLOVE_6B_50D_PATH, GLOVE_840B_300D_PATH)
#print(glove_small)
#print(glove_big)

logging.info(msg='data_size=%s\tCounter(Y)=%s' % (len(Y), Counter(Y)))

model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
print(model.index2word)

w2v = {
    w: vec
    for w, vec in zip(model.index2word, model.syn0)
    }
print(w2v)


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array(
            [
                np.mean(
                    [self.word2vec[w] for w in words if w in self.word2vec]
                        or [np.zeros(self.dim)],
                    axis=0
                )
                for words in X
            ]
        )

# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()]
        )
        return self

    def transform(self, X):
        return np.array([
                np.mean
                (
                    [self.word2vec[w] * self.word2weight[w] for w in words if w in self.word2vec]
                        or [np.zeros(self.dim)],
                    axis=0
                )
                for words in X
            ])


# etree glove small,
# Pipeline: MeanEmbeddingVectorizer -> ExtraTreesClassifier and TfidfEmbeddingVectorizer -> ExtraTreesClassifier
etree_glove_small = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_small)),
                              ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_small_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)),
                                    ("extra trees", ExtraTreesClassifier(n_estimators=200))])


# etree glove big,
# Pipelien: 
etree_glove_big = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_big)),
                            ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_big_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_big)),
                                  ("extra trees", ExtraTreesClassifier(n_estimators=200))])


# etree word2vec
etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                      ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                            ("extra trees", ExtraTreesClassifier(n_estimators=200))])
all_models = [
    ("glove_small", etree_glove_small),
    ("glove_small_tfidf", etree_glove_small_tfidf),
    ("glove_big", etree_glove_big),
    ("glove_big_tfidf", etree_glove_big),
    ("w2v", etree_w2v),
    ("w2v_tfidf", etree_w2v_tfidf),
]
scores = sorted([(name, cross_val_score(model, X, Y, cv=5).mean())
                 for name, model in all_models],
                key=lambda (_, x): -x)
print(tabulate(scores, floatfmt=".4f", headers=("model", 'score')))
