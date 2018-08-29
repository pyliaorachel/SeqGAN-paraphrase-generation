import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init

from .utils import helpers


class Attention(nn.Module):
    """
    Copied & modified from https://pytorch-nlp-tutorial-ny2018.readthedocs.io/en/latest/day2/patterns/attention.html
    """
    def __init__(self, attention_size):
        super().__init__()
        self.attention = self.new_parameter(attention_size * 2, 1) # (hidden_dim * 2) x 1

    def new_parameter(self, *size):
        out = nn.Parameter(torch.Tensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, context, h):
        """
        Inputs:
            - context: batch_size x inp_seq_len x hidden_dim
            - h: batch_size x hidden_dim
        """
        # Concatenate last hidden state with the context vector at each timestep
        batch_size, inp_seq_len, hidden_dim = context.shape
        h = h.unsqueeze(1).repeat(1, inp_seq_len, 1)
        combined = torch.cat([context, h], dim=-1) # batch_size x inp_seq_len x (hidden_dim * 2)

        # Compute attention score
        attn_score = torch.matmul(combined, self.attention).squeeze(-1) # batch_size x inp_seq_len
        attn_score = torch.softmax(attn_score, dim=1).unsqueeze(-1)     # batch_size x inp_seq_len x 1

        # Apply scores as weights & sum across timesteps
        scored_inp = context * attn_score                               # batch_size x inp_seq_len x hidden_dim
        feature = torch.sum(scored_inp, dim=1)                          # batch_size x hidden_dim

        return feature

class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, word_emb, max_seq_len=30, start_token=2, end_token=1, pad_token=0, gpu=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.word_emb = word_emb
        self.gpu = gpu
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token

        self.embeddings = self.init_emb()
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.gru2out = nn.Linear(hidden_dim, word_emb.vocab_size)
        self.attn = Attention(hidden_dim)

        if gpu:
            self.cuda()
            self.attn.cuda()

    def init_emb(self, trainable=True):
        emb_layer = nn.Embedding(self.word_emb.vocab_size, self.embedding_dim)
        emb_layer.load_state_dict({ 'weight': self.word_emb.emb })

        if not trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer

    def init_hidden(self, batch_size=1, gpu=False):
        h = torch.zeros(1, batch_size, self.hidden_dim) # 1 for num_layers * num_directions
        return h.cuda() if gpu else h

    def init_context(self, batch_size, inp_seq_len, gpu=False):
        context = torch.zeros(batch_size, inp_seq_len, self.hidden_dim)
        return context.cuda() if gpu else context

    def forward(self, inp, hidden, context=None):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1).
        When context is provided, the generator is in decoding stage and needs attention.

        Inputs:
            - inp: batch_size x seq_len (1)
            - hidden: 1 x batch_size x hidden_dim
            - context: batch_size x encoded_inp_seq_len x hidden_dim
        """
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(-1, 1, self.embedding_dim)               # batch_size x 1 x embedding_dim

        if context is not None:
            hidden = self.attn(context, hidden[0]).unsqueeze(0)

        out, hidden = self.gru(emb, hidden)                     # batch_size x 1 x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def start_token_tensor(self, batch_size, gpu=False):
        """
        Return start token as tensor with the specified batch size.
        """
        start_tokens = torch.ones(batch_size, 1).long() * self.start_token
        return start_tokens.cuda() if gpu else start_tokens

    def sample(self, cond, gpu=False):
        """
        Samples the network, returns num_samples samples of length max_seq_len, wrapped in variables.
        max_seq_len is the max length among all samples; samples shorter than max_seq_len are padded.

        Inputs:
            - cond: num_samples x max_seq_len
        Outputs: samples, lens
            - samples: num_samples x max_seq_len
            - lens: num_samples
        """
        num_samples = cond.shape[0]

        out, h, context = self.encode(cond, gpu=gpu)
        start_tokens = self.start_token_tensor(num_samples, gpu=gpu)
        samples, lens = self.continue_sample_N(start_tokens, 1, h, context, gpu=gpu)
        return samples.view((num_samples, -1)), lens.view(-1) # samples: remove second dimension used for N; lens: flatten

    def sample_until_end(self, cond, max_len):
        """
        Samples one sentence based on cond until end token is generated.

        Inputs:
            - cond: seq_len
        Outputs: sample
            - sample: seq_len
        """
        cond = cond.unsqueeze(0)

        # Encode
        out, h, context = self.encode(cond)

        # Keep sampling until end token generated, starting from start token
        out = self.start_token_tensor(1).view(-1)
        sample = out
        has_ended = False
        l = 1
        while not has_ended and l < max_len:
            out, h = self.forward(out, h, context)
            out = self.sample_one(out)

            sample = torch.cat([sample, out])
            has_ended = out == self.end_token

            l += 1

        return sample

    def continue_sample_N(self, inp, n, h=None, context=None, gpu=False):
        """
        Continue to sample given some previous subsequences. Repeated n times.
        Returns n samples of length max_seq_len for each inp subsequence.
        max_seq_len is the max length among all samples; samples shorter than max_seq_len are padded.

        Inputs:
            - inp: batch_size x sub_seq_len
            - h: num_layers * num_directions x batch_size x hidden_dim
            - context: batch_size x sub_seq_len x hidden_dim

        Outputs: samples, lens
            - samples: batch_size x n x max_seq_len
            - lens: batch_size x n
        """
        if self.gpu and not gpu:
            self.cpu()

        batch_size, sub_seq_len = inp.size()
        samples = torch.ones(n * batch_size, self.max_seq_len).long() * self.pad_token

        if gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        # If raw sequence is given, encode it first
        if h is None:
            out, h, context = self.encode(inp, gpu=gpu)

        # Init rollout stuffs
        more_samples = torch.LongTensor()                      # for keeping subsequence sequences
        l = sub_seq_len                                        # current length
        lens = torch.ones(n * batch_size).long() * sub_seq_len # init lengths of all sequences
        has_ended = torch.zeros(n * batch_size).byte()         # which sequences has observed end token

        # Monte Carlo search: continue sampling until the end for n times
        samples[:, :sub_seq_len] = inp.unsqueeze(1).repeat(1, 1, n).view(n * batch_size, -1) # copy inputs to samples
        h = h.unsqueeze(1).repeat(1, 1, 1, n).view(1, -1, self.hidden_dim)                   # repeat each hidden vec n times
        out = inp[:, -1].unsqueeze(1).repeat(1, n).view(-1)                                  # repeat each last timestep n times

        if gpu:
            has_ended = has_ended.cuda()
            lens = lens.cuda()
            more_samples = more_samples.cuda()

        while not has_ended.all() and l < self.max_seq_len:             # end if all sequences ended, or exceeded max len
            out, h = self.forward(out, h, context)
            out = self.sample_one(out)

            out[has_ended] = self.pad_token                             # pad ended sequences
            more_samples = torch.cat([more_samples, out.unsqueeze(0)])  # append new samples
            lens[has_ended ^ 1] += 1                                    # update lengths of not-ended sequences
            has_ended[out == self.end_token] = 1                        # update ended sequences

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

        if self.gpu and not gpu:
            self.cuda()

        return samples, lens

    def rollout(self, inp, inp_lens, cond, cond_lens, rollout_num, gpu=False):
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
        if self.gpu and not gpu:
            self.cpu()

        # Encode cond
        out, h, context = self.encode(cond, gpu=gpu)

        # Rollout
        batch_size, seq_len = inp.shape
        rollout_targets = torch.ones(seq_len - 1, batch_size, rollout_num, self.max_seq_len).long() * self.pad_token
        rollout_target_lens = torch.zeros(seq_len - 1, batch_size, rollout_num).long()

        if gpu:
            rollout_targets = rollout_targets.cuda()
            rollout_target_lens = rollout_target_lens.cuda()

        for t in range(seq_len-1):
            out, h = self.forward(inp[:, t], h, context)
            samples, lens = self.continue_sample_N(inp[:, :t+1], rollout_num, h, gpu=gpu)
            lens = lens.view(batch_size, -1)

            samples_seq_len = samples.shape[-1]
            rollout_targets[t, :, :, :samples_seq_len], rollout_target_lens[t, :] = samples, lens
        
        # Repeat conditions
        rollout_cond = cond.view(1, batch_size, 1, -1) \
                           .repeat((seq_len - 1), 1, 1, rollout_num) \
                           .view((seq_len - 1), batch_size, rollout_num, -1)
        rollout_cond_lens = cond_lens.view(1, batch_size, -1) \
                                .repeat((seq_len - 1), 1, rollout_num) \
                                .view((seq_len - 1), batch_size, rollout_num)

        if gpu:
            rollout_cond = rollout_cond.cuda()
            rollout_cond_lens = rollout_cond_lens.cuda()

        if self.gpu and not gpu:
            self.cuda()

        return rollout_targets, rollout_target_lens, rollout_cond, rollout_cond_lens

    def encode(self, inp, gpu=False):
        """
        Encodes the given input. Returns the last output, last hidden state, and historical hidden states (context).

        Inputs:
            - inp: batch_size x inp_seq_len
        Outputs: out, h, context
            - out: batch_size x vocab_size (output for last timestep)
            - h: num_layers * num_directions x batch_size x hidden_dim
            - context: batch_size x inp_seq_len x hidden_dim
        """
        # TODO: move to another encoder class, with padding considered
        if self.gpu and not gpu:
            self.cpu()

        batch_size, inp_seq_len = inp.shape
        inp = inp.t()           # inp_seq_len x batch_size
        h = self.init_hidden(batch_size, gpu=gpu)
        context = self.init_context(batch_size, inp_seq_len, gpu=gpu)

        for i in range(inp_seq_len):
            out, h = self.forward(inp[i], h)
            context[:, i] = h[0] # remove first dimension of num_directions x num_layers

        if self.gpu and not gpu:
            self.cuda()

        return out, h, context

    def sample_one(self, out):
        """
        Sample one batch from distribution.

        Inputs:
            - out: batch_size x vocab_size
        """
        # Sampling from each row; exp to turn log_softmax back
        return torch.multinomial(torch.exp(out), 1).view(-1) 

    def batchNLLLoss(self, inp, inp_lens, target, target_lens, teacher_forcing_ratio=0, gpu=False):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs:
            - inp: batch_size x inp_seq_len
            - inp_lens: batch_size
            - target: batch_size x target_seq_len
            - target_lens: batch_size
        """
        if self.gpu and not gpu:
            self.cpu()

        loss_fn = nn.NLLLoss()
        batch_size, target_seq_len = target.size()

        # Encode inp
        out, h, context = self.encode(inp, gpu=gpu)

        # Decode to output, accumulate loss
        target = target.t()                         # target_seq_len x batch_size
        loss = 0

        # Scheduled sampling
        last_out = self.start_token_tensor(batch_size, gpu=gpu)
        for i in range(1, target_seq_len):
            out, h = self.forward(last_out, h, context)
            loss += loss_fn(out, target[i])

            teacher_forcing = np.random.uniform() < teacher_forcing_ratio
            last_out = target[i] if teacher_forcing else self.sample_one(out)

        if self.gpu and not gpu:
            self.cuda()

        return loss # per batch

    def batchPGLoss(self, inp, target, rewards, gpu=False):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs:
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - rewards: seq_len x batch_size (discriminator reward for each token in the sentence)
        """
        if self.gpu and not gpu:
            self.cpu()

        batch_size, seq_len = target.shape
        target = target.t()     # seq_len x batch_size
        h = self.init_hidden(batch_size, gpu=gpu)

        # Rescale rewards: https://arxiv.org/pdf/1709.08624.pdf
        _, sort_idx = torch.sort(rewards, dim=1, descending=True)
        all_idx, _ = torch.sort(sort_idx, dim=1)
        all_idx = all_idx.float() + 1
        rankings = torch.zeros_like(rewards).scatter_(-1, sort_idx, all_idx)
        rewards = torch.sigmoid(12.0 * (0.5 - rankings / batch_size))

        # Encode inp
        out, h, context = self.encode(inp, gpu=gpu) # out is log_softmax

        # Accumulate loss: log(P(y_t | Y_1:Y_{t-1})) * Q
        log_probs = torch.gather(out, -1, target[0].unsqueeze(1)).view(-1)
        loss = -torch.sum(log_probs * rewards[0])
        for i in range(seq_len-1):
            out, h = self.forward(target[i], h, context)

            log_probs = torch.gather(out, -1, target.data[i+1].unsqueeze(1)).view(-1)
            loss += -torch.sum(log_probs * rewards[i+1])

        if self.gpu and not gpu:
            self.cuda()

        return loss / batch_size

    def turn_on_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def turn_off_grads(self):
        for param in self.parameters():
            param.requires_grad = False
