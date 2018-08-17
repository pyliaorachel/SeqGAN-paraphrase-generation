"""
Sample usage:
    loader = DataLoader('dataset/quora_duplicate_questions.tsv')
    loader.load()

    num_samples = 10
    cond_samples, pos_samples, cond_lens, pos_lens, end_of_dataset = loader.sample(num_samples)
"""
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import nltk
import torch.nn.init as init

from . import helpers


class DataLoader:
    def __init__(self, filepath, end_token_str='<E>', pad_token_str='<P>', gpu=False, light_ver=False):
        self.filepath = filepath
        self.gpu = gpu
        self.light_ver = light_ver # Smaller dataset
        self.end_token_str = end_token_str
        self.pad_token_str = pad_token_str

        self.light_size = 1000
        self.end_token = 1
        self.pad_token = 0
        self.cond_samples = None
        self.pos_samples = None
        self.cond_lens = None
        self.pos_lens = None
        self.word_to_int = None
        self.int_to_word = None
        self.vocab = None
        self.max_seq_len = 0

        self.fetcher = self.fetch()
        self.frozen = None 

    @property
    def total_samples(self):
        return len(self.pos_samples) if self.pos_samples is not None else None

    def load(self):
        """
        Load data from dataset. Stores input and target data (padded), word to index mapping, and vocabulary.
        """
        # Build vocab & mapping
        cond_samples = []
        pos_samples = []
        vocab = [self.end_token_str, self.pad_token_str]
        with open(self.filepath, 'r') as fin:
            fin.readline() # ignore header
            reader = csv.reader(fin, delimiter='\t')
            for row in reader:
                if self.light_ver and len(cond_samples) >= self.light_size:
                    break

                # columns: id, qid1, qid2, question1, question2, is_duplicate
                if row[-1] == '1': # positive examples
                    q1 = nltk.word_tokenize(row[3]) + [self.end_token_str]
                    q2 = nltk.word_tokenize(row[4]) + [self.end_token_str]
                    cond_samples.append(q1)
                    pos_samples.append(q2)
                    vocab += nltk.word_tokenize(row[3]) + nltk.word_tokenize(row[4])

        self.vocab = sorted(list(set(vocab)))
        self.word_to_int = dict((w, i) for i, w in enumerate(self.vocab))
        self.int_to_word = dict((i, w) for i, w in enumerate(self.vocab))

        # Map dataset
        self.cond_samples = [self.sent_to_ints(q) for q in cond_samples]
        self.pos_samples = [self.sent_to_ints(q) for q in pos_samples]

        # Map special tokens
        self.end_token = self.word_to_int[self.end_token_str]
        self.pad_token = self.word_to_int[self.pad_token_str]
        
        # Pad dataset, turn into np array
        self.cond_samples, self.cond_lens = helpers.pad_samples(self.cond_samples, self.pad_token)
        self.pos_samples, self.pos_lens = helpers.pad_samples(self.pos_samples, self.pad_token)
        self.max_seq_len = max(torch.max(self.cond_lens).item(), torch.max(self.pos_lens).item())

    def sample(self, num_samples, is_val=False):
        """
        Samples the dataset, returns num_samples samples of length max_seq_len, wrapped in variables.
        max_seq_len is the max length among all samples; samples shorter than max_seq_len are padded.
        cond_ids are the corresponding condition id for each positive sample.
        If is_val, freeze this sampled batch as validation set until release is called.

        Outputs: pos_samples, pos_lens, cond_ids, end_of_dataset
            - pos_samples: num_samples_fetched x seq_len
            - pos_lens: num_samples_fetched
            - cond_ids: num_samples_fetched
        """
        # Fetch next batch
        next(self.fetcher)
        pos_samples, pos_lens, cond_ids, end_of_dataset = self.fetcher.send(num_samples)

        # Freeze samples if is validation set
        if is_val:
            self.freeze(cond_ids[0], cond_ids[-1])

        # Put to GPU
        if self.gpu:
            pos_samples = pos_samples.cuda()
            pos_lens = pos_lens.cuda()

        # Trim
        pos_samples = helpers.trim_trailing_paddings(pos_samples, pos_lens)

        return pos_samples, pos_lens, cond_ids, end_of_dataset

    def fetch_cond_samples(self, cond_ids):
        """
        Return the batch of cond_samples with cond_ids.

        Inputs:
            - cond_ids: batch_size
        Outputs: cond_samples, cond_lens
            - cond_samples: batch_size x seq_len
            - cond_lens: batch_size
        """
        cond_samples, cond_lens = self.cond_samples[cond_ids], self.cond_lens[cond_ids]

        # Put to GPU
        if self.gpu:
            cond_samples = cond_samples.cuda()
            cond_lens = cond_lens.cuda()
        
        # Trim
        cond_samples = helpers.trim_trailing_paddings(cond_samples, cond_lens)

        return cond_samples, cond_lens

    def sent_to_ints(self, s):
        """
        Return index mappings for a sentence.
        """
        ints = []
        for w in s:
            # Skip if word not found in dict
            if w in self.word_to_int:
                ints += [self.word_to_int[w]]
        return ints

    def ints_to_sent(self, ints):
        """
        Return sentence given list of ints.
        """
        return ' '.join([self.int_to_word[i] for i in ints])

    def fetch(self):
        """
        Samples the dataset, returns num_samples raw samples, starting from the next of last fetched samples.

        Outputs: cond_samples[i:j], pos_samples[i:j], end_of_dataset
            - cond_samples[i:j]: num_samples_fetched x seq_len
            - pos_samples[i:j]: num_samples_fetched x seq_len
        """
        i = 0
        n = len(self.cond_samples)
        while i < n:
            num_samples = yield
            j = min(i + num_samples, n)
            sample_idx = list(range(i, j))
            
            # Check if overlap with frozen batch
            if self.frozen is not None:
                start, end = self.frozen
                if i < start and j > start:
                    remaining_after_end = num_samples - (start - i)
                    j = min(end + 1 + remaining_after_end, n)
                    sample_idx = list(range(i, start)) + list(range(end + 1, j))
                elif i >= start and i <= end:
                    j = min(end + 1 + num_samples, n)
                    sample_idx = list(range(end + 1, j))

            # Fetch
            yield (self.pos_samples[sample_idx], self.pos_lens[sample_idx], sample_idx, j == n)

            # Move pointer
            if j == n: # reset
                i = 0
                self.shuffle()
            else: # continue
                i = j

    def shuffle(self):
        """
        Shuffle all samples except frozen ones.
        """
        num_samples = self.pos_samples.shape[0]
        
        self.frozen = (30, 50)
        if self.frozen is not None:
            start, end = self.frozen
            perm = torch.cat([torch.randperm(start), \
                              torch.LongTensor(list(range(start, end + 1))), \
                              (torch.randperm(num_samples - end - 1) + end + 1)])
        else:
            perm = torch.randperm(num_samples)

        self.pos_samples = self.pos_samples[perm]
        self.pos_lens = self.pos_lens[perm]
        self.cond_samples = self.cond_samples[perm]
        self.cond_lens = self.cond_lens[perm]

    def freeze(self, start, end):
        """
        Freeze samples so that they won't be fetched.
        Only one batch can be frozen at a time.
        """
        self.frozen = (start, end) 

    def release(self):
        """
        Unfreeze samples.
        """
        self.frozen = None
    
    def reset(self):
        self.fetcher = self.fetch()
        self.frozen = None
        self.shuffle()
