import numpy as np
import pandas as pd
import bitermplus as btm
import argparse
from sklearn.metrics import normalized_mutual_info_score
import pickle

def __save_pickle(file, path):
    with open(path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def __get_data(path:str, encoding:str) -> pd.DataFrame :
    return pd.read_csv(path, encoding=encoding)

def __run_btm(corpus, labels, seed, num_of_topics, iterations):
    print('preparing data...')
    X, vocabulary, vocab_dict = btm.get_words_freqs(corpus)
    tf = np.array(X.sum(axis=0)).ravel()

    # Vectorizing documents
    docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    docs_lens = list(map(len, docs_vec))
    # Generating biterms
    biterms = btm.get_biterms(docs_vec)

    print('running model...')
    # INITIALIZING AND RUNNING MODEL
    model = btm.BTM(X, vocabulary, seed=12321, T=num_of_topics, M=10, alpha=50/8, beta=0.01)
    model.fit(biterms, iterations=iterations)
    #Now, we will calculate documents vs topics probability matrix (make an inference).
    p_zd = model.transform(docs_vec)

    # Get index of max probability for each document
    top_prob = [np.argmax(i) for i in p_zd]

    print('*****************************')
    print('Evaluating model performance:')
    print('NMI : {}'.format(normalized_mutual_info_score(labels, top_prob)))
    print('*****************************')
    print('savin results...')
    _save_pickle(p_zd, 'btm_result.pickle')
    print('saving model...')
    _save_pickle(model, 'btm_model.pickle')

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run btm model')
    parser.add_argument('--data', help='path to dataset', nargs='?', default='./data/new_dataset.csv', type=str)
    parser.add_argument('--num_of_topics', help='number of topics', nargs='?', default=11, type=int)
    parser.add_argument('--seed', nargs='?', default=12321, type=int)
    parser.add_argument('--M', nargs='?', default=10, type=int)
    parser.add_argument('--alpha', nargs='?', default=50/8, type=float)
    parser.add_argument('--beta', nargs='?', default=0.01, type=float)
    parser.add_argument('--iterations', nargs='?', default=20, type=int)
    parser.add_argument('--encoding', help='encoding to read dataset', nargs='?', default='utf-8', type=str)

    args = parser.parse_args()

    data = __get_data(args.data, args.encoding)
    __run_btm(
        corpus=data['processed_text'].str.strip().tolist(),
        labels=data['topic'],
        seed=args.seed,
        num_of_topics=args.num_of_topics,
        iterations=args.iterations)




