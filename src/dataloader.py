"""
Sample usage:
    loader = DataLoader('dataset/quora_duplicate_questions.tsv')
    loader.load()

    num_samples = 10
    dataX, dataY, X_lens, Y_lens, end_of_dataset = loader.sample(num_samples)
"""
import csv

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import nltk
import torch.nn.init as init

from . import helpers


class DataLoader:
    def __init__(self, filepath, start_token='<S>', end_token='<E>', pad_token='<P>', gpu=False):
        self.filepath = filepath
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.gpu = gpu

        self.dataX = None
        self.dataY = None
        self.word_to_int = None
        self.int_to_word = None
        self.vocab = None

        self.fetcher = self.fetch()

    def load(self):
        """
        Load data from dataset. Stores input and target data (padded), word to index mapping, and vocabulary.
        """
        # Build vocab & mapping
        dataX = []
        dataY = []
        vocab = [self.start_token, self.end_token, self.pad_token]
        with open(self.filepath, 'r') as fin:
            fin.readline() # ignore header
            reader = csv.reader(fin, delimiter='\t')
            for row in reader:
                # columns: id, qid1, qid2, question1, question2, is_duplicate
                if row[-1] == '1': # positive examples
                    q1 = [self.start_token] + nltk.word_tokenize(row[3]) + [self.end_token]
                    q2 = [self.start_token] + nltk.word_tokenize(row[4]) + [self.end_token]
                    dataX.append(q1)
                    dataY.append(q2)
                    vocab += nltk.word_tokenize(row[3]) + nltk.word_tokenize(row[4])

        self.vocab = sorted(list(set(vocab)))
        self.word_to_int = dict((w, i) for i, w in enumerate(self.vocab))
        self.int_to_word = dict((i, w) for i, w in enumerate(self.vocab))

        # Map dataset
        self.dataX = [self.sent_to_ints(q) for q in dataX]
        self.dataY = [self.sent_to_ints(q) for q in dataY]

    def sample(self, num_samples):
        """
        Samples the dataset, returns num_samples samples of length max_seq_len, wrapped in variables.
        max_seq_len is the max length among all samples; samples shorter than max_seq_len are padded.

        Outputs: padded_X, padded_Y, X_lens, Y_lens, end_of_dataset
            - padded_X, padded_Y: num_samples_fetched x max_seq_len
            - X_lens, Y_lens: num_samples_fetched
        """
        # Fetch next batch
        next(self.fetcher)
        dataX, dataY, end_of_dataset = self.fetcher.send(num_samples)

        # Pad sequences
        pad_token = self.word_to_int[self.pad_token]
        padded_X, X_lens = helpers.pad_samples(dataX, pad_token)
        padded_Y, Y_lens = helpers.pad_samples(dataY, pad_token)

        # Wrap in variable
        padded_X = autograd.Variable(torch.LongTensor(padded_X))
        padded_Y = autograd.Variable(torch.LongTensor(padded_Y))
        X_lens = autograd.Variable(torch.LongTensor(X_lens))
        Y_lens = autograd.Variable(torch.LongTensor(Y_lens))

        return (padded_X, padded_Y, X_lens, Y_lens, end_of_dataset)

    def sent_to_ints(self, s):
        """
        Return index mappings for a sentence.
        """
        return [self.word_to_int[w] for w in s]

    def fetch(self):
        """
        Samples the dataset, returns num_samples raw samples, starting from the next of last fetched samples.

        Outputs: dataX[i:j], dataY[i:j], end_of_dataset
            - dataX[i:j]: num_samples_fetched x seq_len (variable size)
            - dataY[i:j]: num_samples_fetched x seq_len (variable size)
        """
        i = 0
        n = len(self.dataX)
        while i < n:
            num_samples = yield

            j = min(i + num_samples, n)
            yield (self.dataX[i:j], self.dataY[i:j], j == n)

            if j == n: # reset
                i = 0
                # TODO: shuffle here
            else: # continue
                i = j
