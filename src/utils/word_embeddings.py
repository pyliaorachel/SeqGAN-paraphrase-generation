import pickle

import numpy as np
import torch


class WordEmbeddings:
    def __init__(self, embedding_dim, pretrained_emb_path_prefix=None):
        self.embedding_dim = embedding_dim
        self.pretrained_emb_path = f'{pretrained_emb_path_prefix}.npy' if pretrained_emb_path_prefix is not None else None
        self.pretrained_emb_info_path = f'{pretrained_emb_path_prefix}_info.pkl' if pretrained_emb_path_prefix is not None else None

        self.vocab = None
        self.word_to_int = None
        self.int_to_word = None
        self.emb = None

    @property
    def vocab_size(self):
        return len(self.vocab)

    def create_emb_matrix(self, vocab):
        """
        Create mappings and word embeddings from vocabulary.
        Word embeddings are from pretrained ones or randomly initialized.
        """
        self.vocab = vocab # sorted list with unique words
        vocab_size = len(vocab)

        if self.pretrained_emb_path is not None:
            pre_vectors, pre_word_to_int, pre_int_to_word = self.load_pretrained()
            word_to_vec = { w: pre_vectors[pre_word_to_int[w]] for w in pre_int_to_word }

            self.emb = np.zeros((vocab_size, self.embedding_dim))
            for i, word in enumerate(vocab):
                self.emb[i] = word_to_vec[word] if word in word_to_vec else np.random.normal(scale=0.6, size=(self.embedding_dim,))
        else:
            self.emb = np.random.normal(scale=0.6, size=(vocab_size, self.embedding_dim))

        self.emb = torch.Tensor(self.emb)
        self.word_to_int = dict((w, i) for i, w in enumerate(vocab))
        self.int_to_word = dict((i, w) for i, w in enumerate(vocab))

    def load_pretrained(self):
        """
        Returns pretrained vectors and word_to_int, int_to_word mappings.
        """
        with open(self.pretrained_emb_path, 'rb') as fin:
            vectors = np.load(fin)
        with open(self.pretrained_emb_info_path, 'rb') as fin:
            data = pickle.load(fin)
            word_to_int = data['word_to_int']
            int_to_word = data['int_to_word']

        return vectors, word_to_int, int_to_word
