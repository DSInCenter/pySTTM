from sklearn.metrics import normalized_mutual_info_score
from models.model import AbstractModel
from tools.Dataset import Dataset
import numpy as np
import pandas as pd
import bitermplus as btm
import argparse
import pickle



class BTM(AbstractModel):
    '''
    Bitermplus implements Biterm topic model for short texts 
    introduced by Xiaohui Yan, Jiafeng Guo, Yanyan Lan, and Xueqi Cheng. 
    Actually, it is a cythonized version of BTM. 

    Source code: https://github.com/maximtrp/bitermplus
    
    '''
    
    def __init__(self,
                num_topics: int = 10,
                iterations: int = 20,
                num_top_words: int = 20,
                alpha: float = 1,
                beta: float = 0.01,
                seed: int = 123
        ):
        """
        initialization of BTM

        :param num_topics : int, Number of topics.
        :param iterations : int, Number of iterations for the model fitting process
        :param num_top_words : int, Number of top words for coherence calculation. 
        :param alpha : float, Model parameter.
        :param beta : float,  Model parameter.
        :param seed : int, Random state seed.
    
        see https://bitermplus.readthedocs.io/en/latest/bitermplus.html
        """
        super().__init__()
        self.hyperparameters['num_topics'] = num_topics
        self.hyperparameters['iterations'] = iterations
        self.hyperparameters['num_top_words'] = num_top_words
        self.hyperparameters['alpha'] = alpha
        self.hyperparameters['beta'] = beta
        self.hyperparameters['seed'] = seed
        self.model = None
        self.vocab = None


    def hyperparameters_info(self):
        """
        Returns hyperparameters informations
        """
        return self.hyperparameters


    def set_hyperparameters(self, **kwargs):
        """
        Set model hyperparameters
        """
        super().set_hyperparameters(**kwargs)


    def train_model(self, dataset, hyperparameters=None, top_words=10):
        '''
        Train the model

        :param dataset: Dataset
        :param hyperparameters: dictionary in the form {hyperparameter name: value}
        :param top_words: number of top significant words for each topic (default: 10)
        '''

        if hyperparameters is None:
            hyperparameters = {}
        self.hyperparameters.update(hyperparameters)

        ######################### Need Dataset.texts

        # Obtaining terms frequency in a sparse matrix and corpus vocabulary
        X, vocabulary, vocab_dict = btm.get_words_freqs()

        # Vectorizing documents
        docs_vec = btm.get_vectorized_docs(Dataset.texts, vocabulary)
        # Generating biterms
        biterms = btm.get_biterms(docs_vec)

        # Initializing and running model
        model = btm.BTM(X, vocabulary, seed=12321, T=11, M=10, alpha=50/8, beta=0.01)
        model.fit(biterms, iterations=20)

        #Now, we will calculate documents vs topics probability matrix (make an inference).
        p_zd = model.transform(docs_vec)

        self.model = model


    def _select_words(self, topic_id: int, words_num):
        probs = self.model.matrix_topics_words_[topic_id, :]
        idx = np.argsort(probs)[:-words_num-1:-1]
        result = pd.Series(self.model.vocabulary_[idx])
        result.name = 'topic{}'.format(topic_id)
        return result


    def _get_topics_words(self, words_num=20):
        topics_num = self.model.topics_num_
        topics_idx = np.arange(topics_num)
        top_words_btm = pd.concat(map(lambda x: self._select_words(x, words_num), topics_idx), axis=1)
        return top_words_btm




# def __save_pickle(file, path):
#     with open(path, 'wb') as handle:
#         pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


# def __get_data(path:str, encoding:str) -> pd.DataFrame :
#     return pd.read_csv(path, encoding=encoding)

# def __run_btm(corpus, labels, seed, num_of_topics, iterations):
#     print('preparing data...')
#     X, vocabulary, vocab_dict = btm.get_words_freqs(corpus)
#     tf = np.array(X.sum(axis=0)).ravel()

#     # Vectorizing documents
#     docs_vec = btm.get_vectorized_docs(texts, vocabulary)
#     docs_lens = list(map(len, docs_vec))
#     # Generating biterms
#     biterms = btm.get_biterms(docs_vec)

#     print('running model...')
#     # INITIALIZING AND RUNNING MODEL
#     model = btm.BTM(X, vocabulary, seed=12321, T=num_of_topics, M=10, alpha=50/8, beta=0.01)
#     model.fit(biterms, iterations=iterations)
#     #Now, we will calculate documents vs topics probability matrix (make an inference).
#     p_zd = model.transform(docs_vec)

#     # Get index of max probability for each document
#     top_prob = [np.argmax(i) for i in p_zd]

#     print('*****************************')
#     print('Evaluating model performance:')
#     print('NMI : {}'.format(normalized_mutual_info_score(labels, top_prob)))
#     print('*****************************')
#     print('savin results...')
#     _save_pickle(p_zd, 'btm_result.pickle')
#     print('saving model...')
#     _save_pickle(model, 'btm_model.pickle')

    


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Run btm model')
#     parser.add_argument('--data', help='path to dataset', nargs='?', default='./data/new_dataset.csv', type=str)
#     parser.add_argument('--num_of_topics', help='number of topics', nargs='?', default=11, type=int)
#     parser.add_argument('--seed', nargs='?', default=12321, type=int)
#     parser.add_argument('--M', nargs='?', default=10, type=int)
#     parser.add_argument('--alpha', nargs='?', default=50/8, type=float)
#     parser.add_argument('--beta', nargs='?', default=0.01, type=float)
#     parser.add_argument('--iterations', nargs='?', default=20, type=int)
#     parser.add_argument('--encoding', help='encoding to read dataset', nargs='?', default='utf-8', type=str)

#     args = parser.parse_args()

#     data = __get_data(args.data, args.encoding)
#     __run_btm(
#         corpus=data['processed_text'].str.strip().tolist(),
#         labels=data['topic'],
#         seed=args.seed,
#         num_of_topics=args.num_of_topics,
#         iterations=args.iterations)




