import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init

from . import helpers


class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len=30, end_token=1, pad_token=0, gpu=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu
        self.end_token = end_token
        self.pad_token = pad_token

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, batch_size=1):
        h = Variable(torch.zeros(1, batch_size, self.hidden_dim)) # 1 for num_layers * num_directions
        return h.cuda() if self.gpu else h

    def forward(self, inp, hidden):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)

        Inputs:
            - inp: batch_size x seq_len (1)
            - hidden: 1 x batch_size x hidden_dim
        """
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(-1, 1, self.embedding_dim)               # batch_size x 1 x embedding_dim
        out, hidden = self.gru(emb, hidden)                     # batch_size x 1 x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, cond):
        """
        Samples the network, returns num_samples samples of length max_seq_len, wrapped in variables.
        max_seq_len is the max length among all samples; samples shorter than max_seq_len are padded.

        Inputs:
            - cond: num_samples x max_seq_len
        Outputs: samples, lens
            - samples: num_samples x max_seq_len_s
            - lens: num_samples
        """
        num_samples = cond.shape[0]

        out, h = self.encode(cond)
        out = self.sample_one(out).view(-1, 1)
        samples, lens = self.continue_sample_N(out, 1, h)
        return samples.view((num_samples, -1)), lens.view(-1) # samples: remove second dimension used for N; lens: flatten

    def continue_sample_N(self, inp, n, h=None):
        """
        Continue to sample given some previous subsequences. Repeated n times.
        Returns n samples of length max_seq_len for each inp subsequence.
        max_seq_len is the max length among all samples; samples shorter than max_seq_len are padded.

        Inputs:
            - inp: batch_size x sub_seq_len
            - h: num_layers * num_directions, batch_size, hidden_dim

        Outputs: samples, lens
            - samples: batch_size x n x max_seq_len
            - lens: batch_size x n
        """
        batch_size, sub_seq_len = inp.size()
        samples = torch.ones(n * batch_size, self.max_seq_len).long() * self.pad_token
        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        # If raw sequence is given, encode it first, init rollout stuffs correspondingly
        if h is None: # not encoded, encode input which generates the first output
            out, h = self.encode(inp)
            out = self.sample_one(out)

            more_samples = out.unsqueeze(0) # for keeping subsequence sequences, starting from last output 
            l = sub_seq_len + 1             # already one output following the original subsequence
            lens = torch.ones(n * batch_size).long() * (sub_seq_len + 1)        # init lengths
            has_ended = torch.zeros(n * batch_size).byte()                      # which sequences has observed end token 
            has_ended[out == self.end_token] = 1
        else: # already encoded, take last input token as last output
            out = inp[:, -1]

            more_samples = torch.LongTensor()
            l = sub_seq_len
            lens = torch.ones(n * batch_size).long() * (sub_seq_len)
            has_ended = torch.zeros(n * batch_size).byte()

        # Monte Carlo search: continue sampling until the end for n times
        samples[:, :sub_seq_len] = inp.unsqueeze(1).repeat(1, 1, n).view(n * batch_size, -1) # copy inputs to samples
        h = h.unsqueeze(1).repeat(1, 1, 1, n).view(1, -1, self.hidden_dim)                   # repeat each hidden vec n times
        out = out.unsqueeze(1).repeat(1, n).view(-1)                                         # repeat each last timestep n times

        if self.gpu:
            has_ended = has_ended.cuda()
            lens = lens.cuda()
            more_samples = more_samples.cuda()

        # TODO: add teacher forcing
        while not has_ended.all() and l < self.max_seq_len:             # end if all sequences ended, or exceeded max len
            out, h = self.forward(out, h)
            out = self.sample_one(out)

            out[has_ended] = self.pad_token                             # pad ended sequences
            more_samples = torch.cat([more_samples, out.unsqueeze(0)])  # append new samples
            has_ended[out == self.end_token] = 1                        # update ended sequences
            lens[has_ended ^ 1] += 1                                    # update lengths of not-ended sequences

            l += 1

        # Append to original subsequence
        more_samples = more_samples.t()
        more_samples[has_ended ^ 1, -1] = self.end_token                # force sequences to end
        more_seq_len = more_samples.shape[1]
        samples[:, sub_seq_len:sub_seq_len+more_seq_len] = more_samples # concat to original subsequence
        
        # Trim
        samples = helpers.trim_trailing_paddings(samples, lens)

        # Reshape
        lens = lens.view(batch_size, n, -1)
        samples = samples.view(batch_size, n, -1)
        return samples, lens

    def rollout(self, inp, inp_lens, cond, cond_lens, rollout_num):
        """
        Rollout rollout_num times for each timestep of inp, based on this generator's policy.
        Returns all rollout sequences.

        Inputs:
            - inp, cond: batch_size x seq_len
            - inp_lens, cond_lens: batch_size

        Outputs: rollout_targets, rollout_target_lens, rollout_cond, rollout_cond_lens
            - rollout_targets, rollout_cond: (seq_len - 1) x batch_size x rollout_num x seq_len
            - rollout_target_lens, rollout_cond_lens: (seq_len - 1) x batch_size x rollout_num x seq_len
        """
        # Encode cond
        out, h = self.encode(cond)

        # Rollout
        batch_size, seq_len = inp.shape
        rollout_targets = torch.ones(seq_len - 1, batch_size, rollout_num, self.max_seq_len).long() * self.pad_token
        rollout_target_lens = torch.zeros(seq_len - 1, batch_size, rollout_num).long()

        for t in range(seq_len-1):
            out, h = self.forward(inp[:, t], h)
            samples, lens = self.continue_sample_N(inp[:, :t+1], rollout_num, h)
            lens = lens.view(batch_size, -1)
            rollout_targets[t, :], rollout_target_lens[t, :] = samples, lens 
        
        # Repeat conditions
        rollout_cond = cond.view(1, batch_size, 1, -1) \
                           .repeat((seq_len - 1), 1, 1, rollout_num) \
                           .view((seq_len - 1), batch_size, rollout_num, -1)
        rollout_cond_lens = cond_lens.view(1, batch_size, -1) \
                                .repeat((seq_len - 1), 1, rollout_num) \
                                .view((seq_len - 1), batch_size, rollout_num)

        return rollout_targets, rollout_target_lens, rollout_cond, rollout_cond_lens

    def encode(self, inp):
        """
        Encode the given input.

        Inputs:
            - inp: batch_size x inp_seq_len
        Outputs: out, h
            - out: batch_size x vocab_size (output for last timestep)
            - h: num_layers * num_directions, batch_size, hidden_dim
        """
        # TODO: move to another encoder class, with padding considered
        batch_size, inp_seq_len = inp.shape
        inp = inp.t()           # inp_seq_len x batch_size
        h = self.init_hidden(batch_size)

        for i in range(inp_seq_len):
            out, h = self.forward(inp[i], h)
        
        return out, h

    def sample_one(self, out):
        """
        Sample one batch from distribution.

        Inputs:
            - out: batch_size x vocab_size
        """
        # Sampling from each row; exp to turn log_softmax back
        return torch.multinomial(torch.exp(out), 1).view(-1) 

    def batchNLLLoss(self, inp, inp_lens, target, target_lens, teacher_forcing_ratio=0.6):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs:
            - inp: batch_size x inp_seq_len
            - inp_lens: batch_size
            - target: batch_size x target_seq_len
            - target_lens: batch_size
        """
        loss_fn = nn.NLLLoss()
        batch_size, target_seq_len = target.size()

        # Encode inp
        out, h = self.encode(inp)

        # Decode to output, accumulate loss
        target = target.t()                         # target_seq_len x batch_size
        loss = loss_fn(out, target[0])              # loss init to first output's loss

        teacher_forcing = np.random.uniform() < teacher_forcing_ratio
        if teacher_forcing: # feeds target as input to each decoder step
            for i in range(1, target_seq_len):
                out, h = self.forward(target[i-1], h)
                loss += loss_fn(out, target[i])
        else:               # feeds last output as input to each decoder step
            last_out = self.sample_one(out) 
            for i in range(1, target_seq_len):
                out, h = self.forward(last_out, h)
                loss += loss_fn(out, target[i])
                last_out = self.sample_one(out) 

        return loss # per batch

    def batchPGLoss(self, inp, target, rewards):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs:
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - rewards: seq_len x batch_size (discriminator reward for each token in the sentence)

            inp should be target with <s> (start letter) prepended
        """
        batch_size, seq_len = target.shape
        target = target.t()     # seq_len x batch_size
        h = self.init_hidden(batch_size)

        # Encode inp, get the first output
        out, h = self.encode(inp) # out is log_softmax

        # Accumulate loss: log(P(y_t | Y_1:Y_{t-1})) * Q
        log_probs = torch.gather(out, -1, target[0].unsqueeze(1)).view(-1)
        loss = -torch.sum(log_probs * rewards[0])
        for i in range(seq_len-1):
            out, h = self.forward(target[i], h) 

            log_probs = torch.gather(out, -1, target.data[i+1].unsqueeze(1)).view(-1)
            loss += -torch.sum(log_probs * rewards[i+1])

        return loss / batch_size
