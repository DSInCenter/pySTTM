'''
Topic Modeling with NMF
'''

import itertools
from Dataset.Dataset import Dataset
import numpy as np


class NMF:
    def __init__(self) -> None:
        '''
        NMF topic modeling class.\n
        Attributes:\n
        -------------

        '''
        pass

    def __calculate_tf(self, dataset) -> np.array:
        _tf = np.zeros((len(dataset.train_corpus), len(dataset.vocab))) # initializing term-frequency matrix
        tokenized_docs = [doc.split() for doc in dataset.train_corpus]

        for (i, j) in tokenized_docs:
            for (m, n) in dataset.vocab:
                if n in j:
                    _tf[i][dataset.word_to_index(n)] += np.round(j.count(n) / len(j), 3)
        return _tf

    def __calculate_tfidf(self, dataset, tf:np.array) -> np.array :
        N = len(dataset.train_corpus)
        for i in range(tf.shape[0]):
            for j in range(tf.shape[1]):
                tf[i][j] = tf[i][j] * np.round(np.log((N / dataset.words_count[dataset.index_to_word[j]])), 3)
        return tf
    
    def fit(self, dataset:Dataset, iter:int, stop:float) -> None:
        self.tfidf = self.__calculate_tfidf(dataset, self.__calculate_tf(dataset))
