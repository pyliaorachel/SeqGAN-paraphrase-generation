import argparse

import torch

from src.generator import Generator
from src.dataloader import DataLoader
from src.pathbuilder import PathBuilder
from src.static_params import *


def parse_args():
    parser = argparse.ArgumentParser(description='Paraphrase generation')
    parser.add_argument('model', type=str, metavar='F',
                        help='pretrained model directry path; must end with slash')
    return parser.parse_args()

def generate(gen, oracle):
    cond_str = input('Input a sentence (or \'q\' to exit): ')
    while cond_str != 'q':
        cond = torch.LongTensor(oracle.sent_to_ints(cond_str))

        result = gen.sample_until_end(cond, max_len=100).numpy()
        result_str = oracle.ints_to_sent(result)
        result_str = result_str.replace(oracle.end_token_str, '').strip()

        print(result_str)

        cond_str = input('Input a sentence (or \'q\' to exit): ')

if __name__ == '__main__':
    args = parse_args()
    pb = PathBuilder(path=args.model)

    oracle = DataLoader(dataset_path, end_token_str=END_TOKEN, pad_token_str=PAD_TOKEN, gpu=CUDA, light_ver=LIGHT_VER)
    oracle.load()
    end_token, pad_token, max_seq_len, vocab_size = oracle.end_token, oracle.pad_token, oracle.max_seq_len, len(oracle.vocab) 
    max_seq_len += pb.model_params['gan']['pad'] 

    gen = Generator(pb.model_params['G']['ed'], pb.model_params['G']['hd'], vocab_size,
                    end_token=end_token, pad_token=pad_token, max_seq_len=max_seq_len, gpu=CUDA)

    generate(gen, oracle)
