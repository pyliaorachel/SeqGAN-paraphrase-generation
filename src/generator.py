import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init


class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, oracle_init=False):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

        # initialise oracle network with N(0,1)
        # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
        if oracle_init:
            for p in self.parameters():
                init.normal_(p, 0, 1)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)) # 1 for timesteps

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, inp, hidden):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        # input dim                                             # batch_size
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        out, hidden = self.gru(emb, hidden)                     # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, num_samples, start_letter=0):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """
        inp = autograd.Variable(torch.LongTensor([[start_letter]] * num_samples)) # num_samples x 1
        samples = self.continue_sample_N(inp, 1)
        return samples.reshape((num_samples, -1)) # remove second dimension used for N

    def continue_sample_N(self, inp, n):
        """
        Continue to sample given some previous subsequences. Repeated n times.
        Returns n samples of length max_seq_len for each inp subsequence.

        Inputs: inp
            - inp: batch_size x sub_seq_len

        Outputs: samples, hidden
            - samples: batch_size x n x max_seq_length
        """
        batch_size, sub_seq_len = inp.size()
        sub_seq_len -= 1 # deduce to starting symbol

        samples = torch.zeros(n * batch_size, self.max_seq_len).type(torch.LongTensor)
        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        h = self.init_hidden(batch_size)

        # Encode given subsequences
        inp_t = inp[:, 0].reshape(-1)
        for i in range(sub_seq_len):
            out, h = self.forward(inp_t, h)             # out: batch_size x vocab_size
            out = torch.multinomial(torch.exp(out), 1)  # batch_size x 1 (sampling from each row); log to turn log_softmax back to softmax
            inp_t = out.view(-1)

        # Continue sampling until the end for n times
        samples[:, :sub_seq_len] = inp.unsqueeze(1).repeat(1, 1, n).reshape(n * batch_size, -1)[:, 1:] # copy inputs to samples
                                                                                                       # discard start letter
        h = h.unsqueeze(1).repeat(1, 1, 1, n).reshape(1, -1, self.hidden_dim) # repeat each hidden vec n times
        inp_t = inp_t.unsqueeze(1).repeat(1, n).reshape(-1)                   # repeat each last timestep n times
        for i in range(sub_seq_len, self.max_seq_len):
            out, h = self.forward(inp_t, h)
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.data[:, 0]

            inp_t = out.view(-1)

        samples = samples.reshape(batch_size, n, -1)
        return samples

    def rollout(self, inp, rollout_num):
        """
        Rollout rollout_num times for each timestep of inp, based on this generator's policy.
        Returns all rollout sequences.

        Inputs: inp, rollout_num
            - inp: batch_size x seq_len

        Outputs: rollout_targets
            - rollout_targets: (seq_len - 1) x batch_size x rollout_num x seq_len
        """
        batch_size, seq_len = inp.size()
        rollout_targets = torch.zeros(seq_len - 1, batch_size, rollout_num, self.max_seq_len).type(torch.LongTensor)

        for t in range(2, seq_len+1):
            rollout_targets[t-2, :] = self.continue_sample_N(inp[:, :t], rollout_num)

        return rollout_targets

    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += loss_fn(out, target[i])

        return loss     # per batch

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: seq_len x batch_size (discriminator reward for each token in the sentence)

            inp should be target with <s> (start letter) prepended
        """
        batch_size, seq_len = inp.size()
        inp = inp.t()                   # seq_len x batch_size
        target = target.t()             # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h) # out is log_softmax

            # Loss: log(P(y_t | Y_1:Y_{t-1})) * Q
            log_probs = torch.gather(out, -1, target.data[i].unsqueeze(1)).reshape(-1)
            loss += -torch.sum(log_probs * reward[i])

        return loss / batch_size
