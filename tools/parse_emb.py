import argparse
import sys
import os
import pickle
from subprocess import check_output

import numpy as np


# Helper
def line_count(filename):
    return int(check_output(['wc', '-l', filename]).split()[0])

# Parse args
parser = argparse.ArgumentParser(description='Parse word embeddings')
parser.add_argument('filename', metavar='filename', type=str,
                    help='word embedding file')
parser.add_argument('output', metavar='output', type=str,
                    help='output word embedding file, i.e. the actual vectors, in numpy format')
parser.add_argument('output_info', metavar='output-info', type=str,
                    help='output word embedding information file, include int_to_word & word_to_int')

args = parser.parse_args()

# Get embedding file information
vocab_size = line_count(args.filename)
dim = int(args.filename.split('.')[-2][:-1])

# Parse word embedding file
int_to_word = []
idx = 0
word_to_int = {}
vectors = np.empty((vocab_size, dim))
with open(args.filename) as f:
    for line in f:
        line = line.split()
        if len(line) != dim + 1: # space
            continue

        word = line[0]
        int_to_word += [word]
        word_to_int[word] = idx

        vector = line[1:]
        vectors[idx, :] = vector
        
        idx += 1

        if idx % 10000 == 0:
            print('.', end='')
            sys.stdout.flush()

    print('\nFinished.')

# Save
with open(args.output, 'wb') as fout:
    np.save(fout, vectors)
with open(args.output_info, 'wb') as fout:
    data = { 'int_to_word': int_to_word, 'word_to_int': word_to_int }
    pickle.dump(data, fout)
