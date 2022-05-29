'''
Dataset Module to load custom datasets
'''

from ast import Str


class Dataset:
    def __init__(self, path:str, encoding:str='utf-8') -> None:
        '''
        This module is designed to help users load their own custom dataset.\n
        Parameters:\n
        ------------
        path : str\n
            path to folder where corpus lies. To check for the file formats in folder check our github samples.\n
        Attributes:\n
        ------------
        train_corpus : list\n
        test_corpus : list\n
        dev_corpus : list\n
        train_labels : list / optional\n
        test_labels : list / optional\n
        dev_labels : list / optional\n
        vocab : list
        '''
        self.path = path
        self.__load_data(path, encoding)
        self.__load_vocab(path, encoding)

    def __initialize_corpus(self, data:dict) -> None:
        self.train_corpus = data['train_corpus']
        self.test_corpus = data['test_corpus']
        self.dev_corpus = data['dev_corpus']

        self.train_labels = data['train_labels']
        self.test_labels = data['test_labels']
        self.dev_labels = data['dev_labels']

    def __load_data(self, path:str, encoding:str) -> None:
        # sourcery skip: raise-specific-error
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
                        data[f'{_slice}_corpus'].append(_[2])
                    except Exception:
                        print(f'{_slice} is not in [train, test, dev]...')
                elif len(_) == 2:
                    try:
                        data[f'{_slice}_corpus'].append(_[0])
                    except Exception:
                        print(f'{_slice} is not in [train, test, dev]...')
                else:
                    raise Exception('data file must have at least 2 and at most 3 columns...')

    def __load_vocab(self, path:str, encoding:str) -> None:
        self.vocab = []
        with open(f'{path}/vocab.txt', 'r', encoding=encoding) as f:
            lines = f.readlines()
            for line in lines:
                _ = line.split()
                self.vocab.append(_[0])

        



