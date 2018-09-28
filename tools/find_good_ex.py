import csv
import heapq
import argparse


# Parse args
parser = argparse.ArgumentParser(description='Find examples with good evaluation scores in an evaluation result file')
parser.add_argument('filename', metavar='filename', type=str,
                    help='evaluation result tsv file')
parser.add_argument('output', metavar='output', type=str,
                    help='output file')
parser.add_argument('-n', metavar='N', type=int, default=10,
                    help='number of examples to find (default: 10)')
parser.add_argument('--metric', metavar='M', type=str, default='meteor',
                    help='metric to be sorted by (default: meteor)')

args = parser.parse_args()

# Iterate through file
samples = []
with open(args.filename, 'r') as fin:
    reader = csv.reader(fin, delimiter='\t')
    next(reader) # skip header row
    for line in reader:
        cond, pos, neg, bleu, meteor = line
        samples += [{ 'cond': cond, 'pos': pos, 'neg': neg, 'bleu': bleu, 'meteor': meteor }]

        if len(samples) > args.n * 10: # extract top n every few samples
            samples = heapq.nlargest(args.n, samples, key=lambda x: x[args.metric])

samples = heapq.nlargest(args.n, samples, key=lambda x: x[args.metric])

# Write to file
with open(args.output, 'w') as fout:
    writer = csv.writer(fout, delimiter='\t')
    writer.writerow(['original (cond)', 'sample (pos)', 'generated (neg)', 'BLEU', 'METEOR'])
    for sample in samples:
        items = []
        for k, v in sample.items():
            items += [v]
        writer.writerow(items)
