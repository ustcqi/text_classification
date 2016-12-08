# -*- coding:utf-8 -*-
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np

def load_data(filename):
    X, Y = [], []
    wnl = WordNetLemmatizer()

    with open(filename, "rb") as filelines:
        for line in filelines:
            label, text = line.split("\t")
            word_list = [
                wnl.lemmatize(str(word).lower()) for word in text.split()
                if word not in stopwords.words('english')
            ]
            # X.append(text.split())
            if label in ['earn', 'acq', 'crude', 'trade']:
                # if label not in ['earn', 'acq']:
                X.append(word_list)
                Y.append(label)

    X, Y = np.array(X), np.array(Y)
    return X, Y