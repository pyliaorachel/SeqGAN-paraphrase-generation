import torch
from math import ceil
import numpy as np

def prepare_generator_batch(oracle, gen, batch_size, gpu=False):
    """
    Returns samples (a batch) sampled from generator. 

    Returns: inp, inp_lens, cond, cond_lens, end_of_dataset
        - inp, cond: (batch_size * 2) x seq_len
        - inp_lens, cond_lens: (batch_size * 2)
        - cond_lens: (batch_size * 2)
    """
    _, _, cond_ids, end_of_dataset = oracle.sample(batch_size)
    batch_size = len(cond_ids) # update actual sampled batch size
    cond, cond_lens = oracle.fetch_cond_samples(cond_ids)
    target, target_lens = gen.sample(cond)

    # Put to GPU
    if gpu:
        target = target.cuda()
        target_lens = target_lens.cuda()
        cond = cond.cuda()
        cond_lens = cond_lens.cuda()

    return target, target_lens, cond, cond_lens, end_of_dataset 

def prepare_discriminator_data(oracle, gen, batch_size, is_val=False, gpu=False):
    """
    Takes positive (target), negative (generator), and condition sample generators/loaders 
    to prepare inp and target samples for discriminator.

    Returns: inp, inp_lens, cond, cond_lens, target, end_of_dataset
        - inp, cond: (batch_size * 2) x seq_len
        - inp_lens, cond_lens: (batch_size * 2)
        - target: (batch_size * 2) (boolean 1/0)
    """
    # Prepare pos, neg, cond samples
    pos_samples, pos_lens, cond_ids, end_of_dataset = oracle.sample(batch_size, is_val=is_val)
    batch_size = len(cond_ids) # update actual sampled batch size
    cond_samples, cond_lens = oracle.fetch_cond_samples(cond_ids)
    neg_samples, neg_lens = gen.sample(cond_samples)

    _, pos_seq_len = pos_samples.shape
    _, neg_seq_len = neg_samples.shape
    _, cond_seq_len = cond_samples.shape

    pad_token = oracle.pad_token

    # Concat
    inp, inp_lens = cat_samples(pos_samples, pos_lens, neg_samples, neg_lens, pad_token)
    cond, cond_lens = cat_samples(cond_samples, cond_lens, cond_samples, cond_lens, pad_token)

    # Construct target
    target = torch.ones(batch_size * 2)
    target[batch_size:] = 0 # first half is pos, second half is neg

    # Shuffle
    perm = torch.randperm(batch_size * 2)
    inp, inp_lens, cond, cond_lens, target = inp[perm], inp_lens[perm], cond[perm], cond_lens[perm], target[perm]

    # Put to GPU
    if gpu:
        inp = inp.cuda()
        inp_lens = inp_lens.cuda()
        cond = cond.cuda()
        cond_lens = cond_lens.cuda()
        target = target.cuda()

    return inp, inp_lens, cond, cond_lens, target, end_of_dataset

def batchwise_sample(gen, num_samples, batch_size):
    """
    NOT USED.
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """
    # TODO: update gen.sample
    samples = []
    for i in range(int(ceil(num_samples / float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]

def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len, start_letter=0, gpu=False):
    """
    NOT USED.
    """
    s = batchwise_sample(gen, num_samples, batch_size)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(s[i:i+batch_size], start_letter, gpu)
        oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
        oracle_nll += oracle_loss.data.item()

    return oracle_nll / (num_samples / batch_size)

def pad_samples(samples, pad_token):
    """
    Pad samples of variable lengths with pad_token.

    Output: padded_samples, lens
        - padded_samples: num_samples x max_seq_len
        - lens: num_samples
    """
    num_samples = len(samples)

    # Get length info
    lens = np.array([len(x) for x in samples])
    max_len = max(lens)

    # Pad
    padded_samples = np.ones((num_samples, max_len), dtype=np.int) * pad_token

    for i, seq in enumerate(samples):
        padded_samples[i, :lens[i]] = seq

    padded_samples = torch.LongTensor(padded_samples)
    lens = torch.LongTensor(lens)
    return padded_samples, lens

def cat_samples(samples_1, lens_1, samples_2, lens_2, pad_token):
    """
    Concat two batches of samples of variable sizes, and pad with pad_token.

    Inputs:
        - samples_1, samples_2: num_samples x seq_len
        - lens_1, lens_2: num_samples
    Outputs: samples, lens
        - samples: num_samples x seq_len
        - lens: num_samples
    """
    num_samples = len(samples_1) # same length for samples_1 and samples_2
    seq_len_1, seq_len_2 = samples_1.shape[1], samples_2.shape[1]
    max_len = max(seq_len_1, seq_len_2)

    # Concat & pad
    samples = np.ones((num_samples * 2, max_len), dtype=np.int) * pad_token
    samples[:num_samples, :seq_len_1] = samples_1
    samples[num_samples:, :seq_len_2] = samples_2
    samples = torch.LongTensor(samples)
    lens = torch.cat([lens_1, lens_2])

    return samples, lens

def trim_trailing_paddings(samples, sample_lens):
    """
    Trim trailing pad_tokens to align all rows with the max sequence.

    Inputs:
        - samples: batch_size x seq_len
        - sample_lens: batch_size
    Outputs: samples
        - samples: batch_size x seq_len
    """
    max_seq_len = torch.max(sample_lens)
    return samples[:, :max_seq_len]

def sort_sample_by_len(samples, lens):
    """
    Sort samples by length in descending order.

    Inputs:
        - samples: batch_size x seq_len
        - lens: batch_size

    Outputs:
        - samples: batch_size x seq_len
        - lens: batch_size
        - sort_idx: batch_size
    """
    lens, sort_idx = torch.sort(lens, descending=True)
    samples = samples[sort_idx]
    return samples, lens, sort_idx
