from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

from . import generator
from . import discriminator
from . import helpers


DEBUG = True

CUDA = False
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20
START_LETTER = 0
BATCH_SIZE = 32
POS_NEG_SAMPLES = 10000
ROLLOUT_NUM = 5
G_PRETRAIN_EPOCHS = 0 if DEBUG else 100
D_PRETRAIN_STEPS = 0 if DEBUG else 50
D_PRETRAIN_EPOCHS = 0 if DEBUG else 3
G_TRAIN_STEPS = 1 if DEBUG else 1
D_TRAIN_STEPS = 1 if DEBUG else 5
D_TRAIN_EPOCHS = 1 if DEBUG else 3
ADV_TRAIN_ITERS = 1 if DEBUG else 50

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

NEW_ORACLE = False
oracle_samples_path = './oracle/oracle_samples.trc'
oracle_state_dict_path = './oracle/oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_gen_path = './pretrained/gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_dis_path = './pretrained/dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'


def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        print(f'epoch {epoch + 1} : ', end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        # sample from generator and compute oracle NLL
        oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)

        print(f' average_train_NLL = {total_loss:.4f}, oracle_sample_NLL = {oracle_loss:.4f}')

def train_generator_PG(gen, gen_opt, oracle, dis, rollout, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)

        rollout_targets = rollout.rollout(inp, ROLLOUT_NUM)
        rollout_targets_shape = rollout_targets.shape

        rollout_rewards = dis.batchClassify(rollout_targets.reshape(-1, MAX_SEQ_LEN)).reshape(rollout_targets_shape[:-1])
        rollout_rewards = torch.mean(rollout_rewards, -1)
        rewards = dis.batchClassify(target).unsqueeze(0)
        total_rewards = torch.cat([rollout_rewards, rewards])

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, total_rewards)
        pg_loss.backward()
        gen_opt.step()

    # sample from generator and compute oracle NLL
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)

    print(f' oracle_sample_NLL = {oracle_loss:.4f}')

def train_discriminator(discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    # generating a small validation set before training (using oracle and generator)
    pos_val = oracle.sample(100)
    neg_val = generator.sample(100)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        for epoch in range(epochs):
            print(f'd-step {d_step + 1} epoch {epoch + 1} : ', end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            val_acc = torch.sum((val_pred > 0.5) == (val_target > 0.5)).data.item() / 200.
            print(f' average_loss = {total_loss:.4f}, train_acc = {total_acc:.4f}, val_acc = {val_acc:.4f}')

# MAIN
if __name__ == '__main__':
    if NEW_ORACLE:
        oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, oracle_init=True)
        oracle_samples = helpers.batchwise_sample(oracle, POS_NEG_SAMPLES, BATCH_SIZE)
        torch.save(oracle.state_dict(), oracle_state_dict_path)
        torch.save(oracle_samples, oracle_samples_path)

    oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    oracle.load_state_dict(torch.load(oracle_state_dict_path))
    oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)
    
    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    rollout = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

    if CUDA:
        oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()
        rollout = rollout.cuda()
        oracle_samples = oracle_samples.cuda()

    # PRETRAIN GENERATOR
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    train_generator_MLE(gen, gen_optimizer, oracle, oracle_samples, G_PRETRAIN_EPOCHS)

    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, D_PRETRAIN_STEPS, D_PRETRAIN_EPOCHS)

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    if not DEBUG:
        oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)
        print(f'\nInitial Oracle Sample Loss : {oracle_loss:.4f}')

    for i in range(ADV_TRAIN_ITERS):
        print(f'\n--------\nITERATION {i + 1}\n--------')
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, oracle, dis, rollout, G_TRAIN_STEPS)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, D_TRAIN_STEPS, D_TRAIN_EPOCHS)
