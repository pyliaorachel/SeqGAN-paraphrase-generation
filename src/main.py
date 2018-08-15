from __future__ import print_function
import argparse
import logging
import time
from math import ceil
import sys
import os
import pdb

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from . import generator
from . import discriminator
from . import dataloader
from . import pathbuilder
from .params import *
from . import helpers


def parse_args():
    parser = argparse.ArgumentParser(description='Train GAN model')
    # TODO: add arguments
    return parser.parse_args()

def train_generator_MLE(gen, gen_opt, oracle, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    gen.train()
    oracle.reset()

    for epoch in range(epochs):
        print(f'epoch {epoch + 1} : ', end='')

        total_loss = 0
        i = 0
        end_of_dataset = False
        while not end_of_dataset: 
            # Sample from oracle
            pos_samples, pos_lens, cond_ids, end_of_dataset = oracle.sample(BATCH_SIZE)
            cond_samples, cond_lens = oracle.fetch_cond_samples(cond_ids)

            # Train
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(cond_samples, cond_lens, pos_samples, pos_lens, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
            loss.backward()
            gen_opt.step()

            # Accumulate loss
            total_loss += loss.item()

            # Log
            if i % ceil(ceil(oracle.total_samples / float(BATCH_SIZE)) / 10.) == 0: # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

            i += 1

        total_loss /= i # loss in each batch is size_averaged, so divide by num of batches is loss per sample
        logging.info(f'[G_MLE] Average_train_NLL = {total_loss:.4f}')

def train_generator_PG(gen, gen_opt, dis, oracle, rollout, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    gen.train()
    oracle.reset()

    total_loss = 0
    i = 0
    end_of_dataset = False
    for batch in range(num_batches):
        if end_of_dataset:
            break

        # Sample from generator
        target, target_lens, cond, cond_lens, end_of_dataset = helpers.prepare_generator_batch(oracle, gen, BATCH_SIZE, gpu=CUDA)

        # MC search for reward estimation 
        rollout_targets, rollout_target_lens, rollout_cond, rollout_cond_lens \
                = rollout.rollout(target, target_lens, cond, cond_lens, ROLLOUT_NUM)
        rollout_cond_shape = rollout_cond.shape
        rollout_targets_shape = rollout_targets.shape

        rollout_rewards = dis.batchClassify(
                              rollout_targets.view(-1, rollout_targets_shape[-1]),
                              rollout_target_lens.view(-1),
                              rollout_cond.view(-1, rollout_cond_shape[-1]),
                              rollout_cond_lens.view(-1)
                          ).view(rollout_targets_shape[:-1])
        rollout_rewards = torch.mean(rollout_rewards, -1)
        rewards = dis.batchClassify(target, target_lens, cond, cond_lens).unsqueeze(0)
        total_rewards = torch.cat([rollout_rewards, rewards])

        gen_opt.zero_grad()
        loss = gen.batchPGLoss(cond, target, total_rewards)
        loss.backward()
        gen_opt.step()

        # Accumulate loss
        total_loss += loss.item()

        i += 1

    total_loss = total_loss / num_batches 
    logging.info(f'[G_PG] Average_train_NLL = {total_loss:.4f}')

def train_discriminator(dis, dis_opt, gen, oracle, d_steps, epochs):
    """
    Training the discriminator on real samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    dis.train()
    oracle.reset()

    # Generate a small validation set before training (using oracle and generator)
    dataset_size = oracle.total_samples
    valid_set_size = int(dataset_size * VALID_SET_SIZE_RATIO)
    val_inp, val_inp_lens, val_cond, val_cond_lens, val_target, end_of_dataset \
            = helpers.prepare_discriminator_data(oracle, gen, valid_set_size, is_val=True, gpu=CUDA)

    # Train discriminator
    for d_step in range(d_steps):
        for epoch in range(epochs):
            print(f'd-step {d_step + 1} epoch {epoch + 1} : ', end='')

            total_loss = 0
            total_acc = 0
            i = 0
            while not end_of_dataset: 
                # Sample
                inp, inp_lens, cond, cond_lens, target, end_of_dataset = helpers.prepare_discriminator_data(oracle, gen, BATCH_SIZE, gpu=CUDA)

                # Train
                dis_opt.zero_grad()
                loss, acc = dis.batchBCELoss(inp, inp_lens, cond, cond_lens, target)
                loss.backward()
                dis_opt.step()

                # Accumulate loss
                total_loss += loss.item()
                total_acc += acc 

                # Log
                if i % ceil(ceil(oracle.total_samples / float(BATCH_SIZE)) / 10.) == 0: # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

                i += 1

            total_loss /= i # loss in each batch is size_averaged, so divide by num of batches is loss per sample
            total_acc /= (i * BATCH_SIZE) # acc is not averaged, so average here

            _, val_acc = dis.batchBCELoss(val_inp, val_inp_lens, val_cond, val_cond_lens, val_target)
            val_acc /= (valid_set_size * 2)
            logging.info(f'[D] Average_loss = {total_loss:.4f}, train_acc = {total_acc:.4f}, val_acc = {val_acc:.4f}')

    # Release validation set
    oracle.release()

# MAIN
if __name__ == '__main__':
    t = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())

    args = parse_args()
    pb = pathbuilder.PathBuilder(model_params, training_params, pretrain_params)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename=f'./log/{t}_{pb.whole_string()}.log')

    '''Create oracle data loader for pos examples, generator & discriminator for adversarial training, and rollout for MC search'''

    oracle = dataloader.DataLoader(dataset_path, end_token_str=END_TOKEN, pad_token_str=PAD_TOKEN, gpu=CUDA, light_ver=LIGHT_VER)
    oracle.load()
    end_token, pad_token, max_seq_len, vocab_size = oracle.end_token, oracle.pad_token, oracle.max_seq_len, len(oracle.vocab) 
    max_seq_len += MAX_SEQ_LEN_PADDING # give room for longer sequences

    gen = generator.Generator(G_ED, G_HD, vocab_size, end_token=end_token, pad_token=pad_token,
                              max_seq_len=max_seq_len, gpu=CUDA)
    dis = discriminator.Discriminator(D_ED, D_HD, vocab_size, end_token=end_token, pad_token=pad_token,
                                      max_seq_len=max_seq_len, gpu=CUDA)
    rollout = generator.Generator(G_ED, G_HD, vocab_size, end_token=end_token, pad_token=pad_token,
                                  max_seq_len=max_seq_len, gpu=CUDA)

    if pb.pretrain:
        gen.load_state_dict(torch.load(pb.model_path('gen')))
        dis.load_state_dict(torch.load(pb.model_path('dis')))

    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    dis_optimizer = optim.Adagrad(dis.parameters())

    if CUDA:
        gen = gen.cuda()
        dis = dis.cuda()
        rollout = rollout.cuda()

    if not pb.pretrain: # only pretrain G and D if fresh new training
        '''Pretrain generator'''

        print('Starting Generator MLE Training...')
        train_generator_MLE(gen, gen_optimizer, oracle, G_PRETRAIN_EPOCHS)

        torch.save(gen.state_dict(), pb.model_pretrain_path('gen'))
        # gen.load_state_dict(torch.load(pb.model_pretrain_path('gen')))

        '''Pretrain discriminator'''

        print('\nStarting Discriminator Training...')
        train_discriminator(dis, dis_optimizer, gen, oracle, D_PRETRAIN_STEPS, D_PRETRAIN_EPOCHS)

        torch.save(dis.state_dict(), pb.model_pretrain_path('dis'))
        # dis.load_state_dict(torch.load(pb.model_pretrain_path('dis')))

    '''Adversarial training'''

    print('\nStarting Adversarial Training...')
    for i in range(ADV_TRAIN_ITERS):
        print(f'\n--------\nITERATION {i + 1}\n--------')

        '''Train generator'''

        print('\nAdversarial Training Generator : ', end='')
        train_generator_PG(gen, gen_optimizer, dis, oracle, rollout, G_TRAIN_STEPS)

        '''Train discriminator'''

        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, gen, oracle, D_TRAIN_STEPS, D_TRAIN_EPOCHS)

        if i % SAVE_MODEL_ITER == 0:
            if pb.pretrain:
                params = { 'gan': { 'iter': SAVE_MODEL_ITER } }
                pb.increment_training_params(params)

            torch.save(gen.state_dict(), pb.model_path('gen'))
            torch.save(dis.state_dict(), pb.model_path('dis'))
