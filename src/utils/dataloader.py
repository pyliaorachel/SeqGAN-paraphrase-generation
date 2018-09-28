"""
Sample usage:
    word_emb = WordEmbeddings(50, 'path/to/pretrained/embeddings')
    loader = DataLoader('dataset/quora_duplicate_questions.tsv', word_emb)
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
    def __init__(self, filepath, word_emb, train_size=53000, test_size=3000, start_token_str='<S>', end_token_str='<E>', pad_token_str='<P>', gpu=False, light_ver=False, mode='train'):
        self.filepath = filepath
        self.word_emb = word_emb
        self.gpu = gpu
        self.light_ver = light_ver # Smaller dataset
        self.start_token_str = start_token_str
        self.end_token_str = end_token_str
        self.pad_token_str = pad_token_str

        self.train_size = train_size if not light_ver else 1000
        self.test_size = test_size if not light_ver else 300
        self.total_samples = self.train_size + self.test_size
        self.mode = mode

        self.start_token = 2
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

    def load(self):
        """
        Load data from dataset. Stores input and target data (padded), word to index mapping, and vocabulary.
        Only keep data based on train/test mode. If mode is 'train', keep the first train_size samples;
        if mode is 'test', keep the test_size samples after the first train_size samples.
        However, all train_size + test_size samples need to be read in order to build the vocabulary information.
        """
        # Build vocab & mapping
        cond_samples = []
        pos_samples = []
        vocab = [self.start_token_str, self.end_token_str, self.pad_token_str]
        with open(self.filepath, 'r', encoding='utf-8') as fin:
            fin.readline() # ignore header
            reader = csv.reader(fin, delimiter='\t')

            for row in reader:
                if len(cond_samples) >= self.total_samples:
                    break

                # columns: id, qid1, qid2, question1, question2, is_duplicate
                if row[-1] == '1': # positive examples
                    q1_str = row[3].lower()
                    q2_str = row[4].lower()
                    q1_tokens = nltk.word_tokenize(q1_str)
                    q2_tokens = nltk.word_tokenize(q2_str)

                    q1 = [self.start_token_str] + q1_tokens + [self.end_token_str]
                    q2 = [self.start_token_str] + q2_tokens + [self.end_token_str]
                    cond_samples.append(q1)
                    pos_samples.append(q2)
                    vocab += q1_tokens + q2_tokens

        self.vocab = sorted(list(set(vocab)))
        self.word_emb.create_emb_matrix(self.vocab)
        self.word_to_int, self.int_to_word = self.word_emb.word_to_int, self.word_emb.int_to_word

        # Keep only train/test set
        start, end = (0, self.train_size) if self.mode == 'train' else (self.train_size, self.train_size + self.test_size)

        # Map dataset
        self.cond_samples = [self.sent_to_ints(q) for q in cond_samples[start:end]]
        self.pos_samples = [self.sent_to_ints(q) for q in pos_samples[start:end]]

        # Map special tokens
        self.start_token = self.word_to_int[self.start_token_str]
        self.end_token = self.word_to_int[self.end_token_str]
        self.pad_token = self.word_to_int[self.pad_token_str]
        
        # Pad dataset, turn into np array
        self.cond_samples, self.cond_lens = helpers.pad_samples(self.cond_samples, self.pad_token)
        self.pos_samples, self.pos_lens = helpers.pad_samples(self.pos_samples, self.pad_token)
        self.max_seq_len = max(torch.max(self.cond_lens).item(), torch.max(self.pos_lens).item())

    def sample(self, num_samples, is_val=False, gpu=False):
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
        if gpu:
            pos_samples = pos_samples.cuda()
            pos_lens = pos_lens.cuda()

        # Trim
        pos_samples = helpers.trim_trailing_paddings(pos_samples, pos_lens)

        return pos_samples, pos_lens, cond_ids, end_of_dataset

    def fetch_cond_samples(self, cond_ids, gpu=False):
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
        if gpu:
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
        Freeze samples so that they won't be fetched. Used for splitting validation set from training set.
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
