import argparse
import csv
import sys

import torch
from nlgeval import NLGEval

from src.generator import Generator
from src.utils.word_embeddings import WordEmbeddings
from src.utils.dataloader import DataLoader
from src.utils.pathbuilder import PathBuilder
from src.utils import helpers
from src.utils.static_params import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train GAN model')
    parser.add_argument('model', type=str, metavar='model',
                        help='model file path')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained generator')
    parser.add_argument('--no-score', dest='no_score', action='store_true',
                        help='disable standard metric evaluation')
    parser.add_argument('--mode', type=str, choices={'test', 'train'}, metavar='mode', default='test',
                        help='generate on train or test set (default: test)')
    return parser.parse_args()

def tensor_to_sent(t, oracle):
    ints = t.numpy()
    ints = [x for x in ints if x != oracle.pad_token]
    sent = oracle.ints_to_sent(ints)
    sent = sent.replace(oracle.end_token_str, '').strip()
    return sent

def evaluate(gen, oracle, output, no_score=False):
    if not no_score:
        n = NLGEval()

    print('Start evaluation...')
    total_samples = oracle.test_size

    with open(output, 'w') as fout:
        writer = csv.writer(fout, delimiter='\t')

        if not no_score:
            writer.writerow(['original (cond)', 'sample (pos)', 'generated (neg)', 'BLEU', 'METEOR'])
            total_bleu = 0
            total_meteor = 0
        else:
            writer.writerow(['original (cond)', 'sample (pos)', 'generated (neg)'])

        i = 0
        end_of_dataset = False
        while not end_of_dataset:
            # Retrieve test sample from test set
            pos, pos_len, cond_id, end_of_dataset = oracle.sample(1, gpu=False)
            cond, cond_len = oracle.fetch_cond_samples(cond_id, gpu=False)
            pos, pos_len, cond, cond_len = pos[0], pos_len[0], cond[0], cond_len[0]

            # Generate paraphrase
            generated = gen.sample_until_end(cond, max_len=100)

            # Turn to string
            pos_str = tensor_to_sent(pos, oracle)
            cond_str = tensor_to_sent(cond, oracle)
            generated_str = tensor_to_sent(generated, oracle)

            # Calculate BLEU score
            if not no_score:
                scores = n.compute_individual_metrics([pos_str], generated_str)
                bleu = scores['Bleu_2']
                meteor = scores['METEOR']

                total_bleu += bleu
                total_meteor += meteor

            # Output to tsv file
            if not no_score:
                writer.writerow([cond_str, pos_str, generated_str, bleu, meteor])
            else:
                writer.writerow([cond_str, pos_str, generated_str])

            i += 1
            if i % int(total_samples / 5) == 0: # Print progress every 5%
                print('.', end='')
                sys.stdout.flush()

        if not no_score:
            avg_bleu = total_bleu / i
            avg_meteor = total_meteor / i
            print('Average BLEU score: {avg_bleu}\tAverage METEOR score: {avg_meteor}')

if __name__ == '__main__':
    args = parse_args()
    pb = PathBuilder(path=args.model)

    word_emb = WordEmbeddings(pb.model_params['G']['ed'])
    oracle = DataLoader(dataset_path, word_emb, pb.train_size, pb.test_size, end_token_str=END_TOKEN, pad_token_str=PAD_TOKEN, gpu=False, light_ver=LIGHT_VER, mode=args.mode)
    oracle.load()
    end_token, pad_token, max_seq_len, vocab_size = oracle.end_token, oracle.pad_token, oracle.max_seq_len, len(oracle.vocab) 
    max_seq_len += pb.model_params['gan']['pad'] 

    gen = Generator(pb.model_params['G']['ed'], pb.model_params['G']['hd'], word_emb,
                    end_token=end_token, pad_token=pad_token, max_seq_len=max_seq_len, gpu=False)

    # Load model to CPU
    if args.pretrained:
        gen.load_state_dict(torch.load(pb.model_pretrain_path('gen'), map_location=lambda storage, loc: storage))
    else:
        gen.load_state_dict(torch.load(pb.model_path('gen'), map_location=lambda storage, loc: storage))

    gen.eval()
    gen.turn_off_grads()

    output = pb.model_eval_output_path(pretrain=args.pretrained, no_score=args.no_score)
    pb.ensure(output)

    evaluate(gen, oracle, output, args.no_score)
