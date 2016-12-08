# -*- coding:utf-8 -*-
import sys
reload(sys)

import os
import logging
from collections import Counter

from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from gensim.models.word2vec import Word2Vec

from data_loader import load_data


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    filename='./result/text_cls.log',
                    filemode='wb+')


class TextClassification:
    def __init__(self, X=None, Y=None):
        self.data_root = os.getcwd()[:-3] + "data/"
        # TRAIN_SET_PATH = "20ng-no-stop.txt"
        # TRAIN_SET_PATH = "r52-all-terms.txt"
        self.TRAIN_SET_PATH = self.data_root + "r8-no-stop.txt"

        self.GLOVE_6B_50D_PATH = self.data_root + "glove.6B.50d.txt"
        self.GLOVE_840B_300D_PATH = self.data_root + "glove.840B.300d.txt"

        self.n_folds = 5

        self.vectorizers = [
            ("count_vectorizer", CountVectorizer(analyzer=lambda x: x)),
            ("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x))
        ]
        self.estimators = [
            ('mlp', MLPClassifier()),
            ('svc', SVC()),
            ('lr', LogisticRegression()),
            ('rf', RandomForestClassifier())
            #('gbdt', GradientBoostingClassifier())
            #('bernoullinb', BernoulliNB())
            #('multinomialnb', MultinomialNB())
        ]
        self.params = dict(
            bernoullinb=dict(bernoullinb__alpha=[0.8, 1.0]),
            multinomialnb=dict(multinomialnb__alpha=[0.8, 1.0]),
            lr=dict(lr__penalty=['l1', 'l2'],
                    lr__C=[0.9, 0.95],
                    lr__tol=[0.005, 0.001]
                    ),
            svc=dict(svc__C=[0.9, 0.95],
                     svc__kernel=['linear', 'rbf'],
                     svc__tol=[0.005, 0.001]
                     ),
            mlp=dict( # mlp__activation=['tanh', 'relu'],
                    # mlp__solver=['sgd', 'adam'],
                    # mlp__alpha=[0.0001, 0.001],
                    mlp__learning_rate=['constant', 'adaptive'],
                    # mlp__learning_rate_init=[0.01, 0.001],
                    mlp__hidden_layer_sizes=[32, 64],
                    mlp__max_iter=[100, 200],
                    mlp__tol=[0.001, 0.005],
                ),
            rf=dict(rf__n_estimators=[10, 20],
                    rf__bootstrap=[True, False],
                    ),
            gbdt=dict(gbdt__n_estimators=[10, 20],
                      gbdt__learning_rate=[0.1, 0.5]
                      )
        )

        self.X = X
        self.Y = Y

    def set_parameters(self, **kwargs):
        """ set the specified parameters which used in TextClassification
        :param kwargs:
        :return:
        """
        pass

    def benchmark(self, vectorizer, estimator, data_train, data_test):
        (X_train, Y_train) = data_train
        (X_test, Y_test) = data_test
        pipe = Pipeline(steps=[vectorizer, estimator])
        grid_search = GridSearchCV(pipe, param_grid=self.params[estimator[0]], cv=self.n_folds, n_jobs=1)
        grid_search.fit(X_train, Y_train)
        logging.info(msg='estimator=%s\tvectorizer=%s\tbest_params=%s' % (estimator[0], vectorizer[0], grid_search.best_params_))
        Y_predict = grid_search.predict(X_test)
        accuracy = accuracy_score(y_true=Y_test, y_pred=Y_predict)
        return accuracy

    def train_models(self, vectorizers, estimators, X, Y):
        sample_sizes_prop = [0.3, 0.5, 0.7, 0.8, 0.9]
        #sample_sizes_prop = [0.2, 0.25]
        table = []
        models_name = []
        for prop in sample_sizes_prop:
            shuffle_split = ShuffleSplit(n_splits=1, train_size=prop, random_state=0)
            X_sample, Y_sample = [], []
            for train_index, test_index in shuffle_split.split(Y):
                X_sample = [X[idx] for idx in train_index]
                Y_sample = [Y[idx] for idx in train_index]
            X_sample, Y_sample = np.array(X_sample), np.array(Y_sample)
            X_train, X_test, Y_train, Y_test = train_test_split(X_sample, Y_sample, test_size=prop, random_state=9)
            training_data = (X_train, Y_train)
            test_data = (X_test, Y_test)
            for vectorizer in vectorizers:
                for estimator in estimators:
                    model_name = estimator[0] + '_' + vectorizer[0]
                    accuracy = round(self.benchmark(vectorizer, estimator, training_data, test_data), 4)
                    table.append({'data_size': np.shape(Y_sample)[0],
                                  'model': model_name,
                                  'accuracy': accuracy
                                  })
                    models_name.append(model_name)
                    print(prop, model_name, np.shape(Y_sample)[0], accuracy)
                    logging.info(msg="proportion=%s\tdata_size=%s\tmodel=%s\taccuracy=%s"
                                     % (prop, np.shape(Y_sample)[0], model_name, accuracy))
        df = pd.DataFrame(table)
        return df, models_name

    def plot_bar_figure(slef, df, models_name):
        plt.figure(figsize=(15, 6)) # width height
        #sns.barplot(x=models_name,y=[accuracy])
        plt.savefig("./result/bar.png")
        plt.close()

    def plot_point_line(self, df, models_name, vec_type):
        models_name = [name for name in models_name if (str(name).find(str(vec_type)) != -1)]
        logging.info(msg='plot %s point line, models name=%s' % (vec_type, models_name))
        plt.figure(figsize=(15, 6))
        fig = sns.pointplot(x='data_size',
                            y='accuracy',
                            hue='model',
                            data=df[df.model.map(lambda x: x in models_name)]
                            )
        sns.set_context("notebook", font_scale=1.0)
        fig.set(ylabel="accuracy")
        fig.set(xlabel="training examples")
        fig.set(title="text classification benchmark")
        plt.legend(loc='upper left', frameon=False)
        figname = './result/' + str(vec_type) + '.png'
        plt.savefig(figname)
        plt.close()


if __name__ == '__main__':
    data_root = os.getcwd()[:-3] + "data/"
    filename = data_root + "r8-no-stop.txt"
    X, Y = load_data(filename)
    logging.info(msg='data_size=%s\tCounter(Y)=%s' % (len(Y), Counter(Y)))

    text_cls = TextClassification(X=X, Y=Y)
    vectorizers = text_cls.vectorizers
    estimators = text_cls.estimators
    df, models_name = text_cls.train_models(vectorizers, estimators, X, Y)
    df.to_csv(path_or_buf='./result/comparison.csv', sep='\t')
    text_cls.plot_point_line(df, models_name, 'tfidf')
    text_cls.plot_point_line(df, models_name, 'count')
