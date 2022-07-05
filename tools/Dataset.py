'''
Dataset Module to load custom datasets
'''

from operator import index
from textwrap import indent
import numpy as np


class Dataset:
    '''
    This module is designed to help users load their own custom dataset.
    '''
    def __init__(self, path:str, encoding:str='utf-8') -> None:
        '''
        initialization of Dataset
        :param path : string, path to the dataset
        :param encoding : string, encoding to read data (default 'utf-8')
        '''

        self.initialize_corpus(
            self.load_data(path, encoding)) # initialize train, test, dev
        self.load_vocab(path, encoding) # get vocabulary
        self.wordtoindex = {word: index for index, word in enumerate(self.vocab)}
        self.indextoword = {index: word for word, index in self.wordtoindex.items()}
        self.count_words()

    def initialize_corpus(self, data:dict) -> None:
        self.train_corpus = data['train_corpus']
        self.test_corpus = data['test_corpus']
        self.dev_corpus = data['dev_corpus']

        self.train_labels = data['train_labels']
        self.test_labels = data['test_labels']
        self.dev_labels = data['dev_labels']

    def load_data(self, path:str, encoding:str) -> None:
        data = {
            'train_corpus' : [],
            'test_corpus' : [],
            'dev_corpus' : [],
            'train_labels' : [],
            'test_labels' : [],
            'dev_labels' : []
        }

        with open(f'{path}/data.tsv', 'r', encoding=encoding) as f:
            lines = f.readlines()
            for line in lines:
                _ = line.split('\t')
                _slice = _[1]
                if len(_) == 3:
                    try:
                        data[f'{_slice}_corpus'].append(_[0])
                        data[f'{_slice}_labels'].append(_[2])
                    except Exception:
                        print(f'{_slice} is not in [train, test, dev]...')
                elif len(_) == 2:
                    try:
                        data[f'{_slice}_corpus'].append(_[0])
                    except Exception:
                        print(f'{_slice} is not in [train, test, dev]...')
                else:
                    raise Exception('data file must have at least 2 and at most 3 columns...')
        return data

    def load_vocab(self, path:str, encoding:str) -> None:
        self.vocab = ['UNK']
        with open(f'{path}/vocab.txt', 'r', encoding=encoding) as f:
            lines = f.readlines()
            for line in lines:
                _ = line.split()
                self.vocab.append(_[0])
    
    def count_words(self):
        self.words_count = {}
        for voc in self.vocab:
            self.words_count[voc] = 0
            for doc in self.train_corpus:
                self.words_count[voc] += doc.split().count(voc)
