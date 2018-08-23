import argparse

import torch
import nltk

from src.generator import Generator
from src.utils.word_embeddings import WordEmbeddings
from src.utils.dataloader import DataLoader
from src.utils.pathbuilder import PathBuilder
from src.utils.static_params import *
from .evaluate import tensor_to_sent


def parse_args():
    parser = argparse.ArgumentParser(description='Paraphrase generation')
    parser.add_argument('model', type=str, metavar='model',
                        help='pretrained model directry path; must end with slash')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained generator')
    parser.add_argument('--mode', type=str, choices={'test', 'train'}, metavar='mode', default='test',
                        help='generate on train or test set (default: test)')
    return parser.parse_args()

def generate(gen, oracle):
    cond_str = input('Input a sentence (or \'q\' to exit): ')
    while cond_str != 'q':
        cond_tokens = nltk.word_tokenize(cond_str.lower()) + [oracle.end_token_str]
        cond = torch.LongTensor(oracle.sent_to_ints(cond_tokens))

        result = gen.sample_until_end(cond, max_len=100)
        result_str = tensor_to_sent(result, oracle)

        print(result_str)

        cond_str = input('Input a sentence (or \'q\' to exit): ')

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

    if args.pretrained:
        gen.load_state_dict(torch.load(pb.model_pretrain_path('gen')))
    else:
        gen.load_state_dict(torch.load(pb.model_path('gen')))

    gen.eval()
    gen.turn_off_grads()

    generate(gen, oracle)
