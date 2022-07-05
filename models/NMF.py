'''
Topic Modeling with NMF
'''

from tools.Dataset import Dataset
import numpy as np


class NMF:
    def __init__(self, num_topics:int, num_iterations:int=20, stop:float=0.001) -> None:
        '''
        Initialize NMF model
        Parameters
        ----------
        :param num_topics: int, number of topics
        :param num_iterations: int, number of maximum iterations (default : 20)
        :param stop: float, stop after reaching the given value (default : 0.001)
        '''
        super().__init__()
        self.hyperparameters = dict()
        self.hyperparameters["num_topics"] = num_topics
        self.hyperparameters["num_iterations"] = num_iterations
        self.hyperparameters["stop"] = stop

    def __calculate_tf(self, dataset) -> np.array:
        _tf = np.zeros((len(dataset.train_corpus), len(dataset.vocab))) # initializing term-frequency matrix
        tokenized_docs = [doc.split() for doc in dataset.train_corpus]

        for i, j in enumerate(tokenized_docs):
            for m, n in enumerate(dataset.vocab):
                if n in j:
                    _tf[i][dataset.word_to_index(n)] += np.round(j.count(n) / len(j), 3)
        return _tf

    def __calculate_tfidf(self, dataset, tf:np.array) -> np.array :
        N = len(dataset.train_corpus)
        for i in range(tf.shape[0]):
            for j in range(tf.shape[1]):
                tf[i][j] = tf[i][j] * np.round(np.log((N / dataset.words_count[dataset.index_to_word[j]])), 3)
        return tf
    
    def __frobenius_norm(self, v:np.array, w:np.array, h:np.array) -> float:
        return np.linalg.norm(
            (v - np.matmul(w, h)), ord='fro'
        )

    def __update_w(self, v:np.array, w:np.array, h:np.array) -> np.array:
        numerator = np.matmul(v, np.transpose(h))
        denomerator = np.matmul(w, np.matmul(h, np.transpose(h)))
        return np.matmul(w, numerator / denomerator)

    def __update_h(self, v:np.array, w:np.array, h:np.array) -> np.array:
        numerator = np.matmul(np.transpose(w), v)
        denomerator = np.matmul(np.transpose(w), np.matmul(h, np.transpose(w)))
        return np.matmul(h, numerator / denomerator)

    def train_model(self, dataset:Dataset) -> np.array:
        '''
        applies NMF for topic modeling on given dataset
        Parameters:\n
        -------------
        dataset: Dataset obj\n
        n_topics: int\n
            number of topics\n
        n_iterations: int\n
            number of iterations at most we want NMF go through\n
        stop: float\n
            convergence condition\n
        Return:\n
        -----------
        np.array : number_of_docs * number_of_topics np.array
        '''
        self.tfidf = self.__calculate_tfidf(dataset, self.__calculate_tf(dataset))

        self.W = np.random.uniform(0, 1, (len(dataset.train_corpus), self.hyperparameters["num_topics"]))
        self.H = np.random.uniform(0, 1, (self.hyperparameters["num_topics"], len(dataset.vocab)))

        for _ in range(self.hyperparameters["num_iterations"]):
            loss = self.__frobenius_norm(self.tfidf, self.W, self.H)
            if loss <= self.hyperparameters["stop"]:
                print('model converged...')
                return self.W
            self.W = self.__update_w(self.tfidf, self.W, self.H)
            self.H = self.__update_h(self.tfidf, self.W, self.H)
        
        print('end of iterations...')
        return self.W
