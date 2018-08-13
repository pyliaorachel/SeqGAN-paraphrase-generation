"""
BiGRU text classifier.
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb

from . import helpers

class Discriminator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len=30, end_token=1, pad_token=0, gpu=False, dropout=0.2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu
        self.end_token = end_token
        self.pad_token = pad_token

        # Inp
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.gru2hidden = nn.Linear(2 * 2 * hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)

        # Condition
        self.gru_cond = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.gru2hidden_cond = nn.Linear(2 * 2 * hidden_dim, hidden_dim)
        self.dropout_linear_cond = nn.Dropout(p=dropout)

        # Shared
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden2out = nn.Linear(hidden_dim * 2, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2 * 2 * 1, batch_size, self.hidden_dim))
        return h.cuda() if self.gpu else h

    def forward(self, inp, inp_lens, cond, cond_lens, hidden, hidden_cond):
        """
        Embeds input and applies GRU one batch at a time.

        Inputs:
            - inp, cond: batch_size x seq_len
            - inp_lens, cond_lens: batch_size
            - hidden: 4 x batch_size x hidden_dim
        """
        # Inp
        inp, inp_lens, sort_idx = helpers.sort_sample_by_len(inp, inp_lens)      # sort

        emb = self.embeddings(inp)                                               # batch_size x seq_len x embedding_dim

        inp = nn.utils.rnn.pack_padded_sequence(inp, inp_lens, batch_first=True)
        _, hidden = self.gru(emb, hidden)                                        # 4 x batch_size x hidden_dim
        inp, _ = nn.utils.rnn.pad_packed_sequence(inp, batch_first=True)

        out = self.gru2hidden(hidden.view(-1, 4 * self.hidden_dim))              # batch_size x (4 * hidden_dim)
        out = torch.tanh(out)
        out = self.dropout_linear(out)

        inp, inp_lens = inp[sort_idx], inp_lens[sort_idx]                        # unsort

        # Cond
        cond, cond_lens, sort_idx_cond = helpers.sort_sample_by_len(cond, cond_lens)

        emb_cond = self.embeddings(cond)

        cond = nn.utils.rnn.pack_padded_sequence(cond, cond_lens, batch_first=True)
        _, hidden_cond = self.gru_cond(emb_cond, hidden_cond)
        cond, _ = nn.utils.rnn.pad_packed_sequence(cond, batch_first=True)

        out_cond = self.gru2hidden_cond(hidden_cond.view(-1, 4 * self.hidden_dim))
        out_cond = torch.tanh(out_cond)
        out_cond = self.dropout_linear_cond(out_cond)

        cond, cond_lens = cond[sort_idx_cond], cond_lens[sort_idx_cond]

        # Combine
        out = torch.cat([out, out_cond], dim=1) # batch_size x (hidden_dim * 2)
        out = self.hidden2out(out)              # batch_size x 1
        out = torch.sigmoid(out)
        
        return out

    def batchClassify(self, inp, inp_lens, cond, cond_lens):
        """
        Classifies a batch of sequences.

        Inputs:
            - inp, cond: batch_size x seq_len
            - inp_lens, cond_lens: batch_size

        Returns: out
            - out: batch_size ([0,1] score)
        """
        batch_size = inp.shape[0]
        h = self.init_hidden(batch_size)
        h_cond = self.init_hidden(batch_size)
        out = self.forward(inp, inp_lens, cond, cond_lens, h, h_cond)
        return out.view(-1)

    def batchBCELoss(self, inp, inp_lens, cond, cond_lens, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs:
            - inp, cond: batch_size x seq_len
            - inp_lens, cond_lens: batch_size
            - target: batch_size (binary 1/0)
        """
        out = self.batchClassify(inp, inp_lens, cond, cond_lens) 
        acc = torch.sum((out > 0.5) == (target > 0.5)).data.item()
        loss = F.binary_cross_entropy(out, target)
        return loss, acc
