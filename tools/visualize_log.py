import argparse
import re
from collections import defaultdict

import matplotlib.pyplot as plt


class Extractor:
    @staticmethod
    def square_bracket(s):
        pattern = re.compile('\[(.*)\]')
        return pattern.findall(s)[0]

    @staticmethod
    def int(s, field):
        pattern = re.compile(f'{field} = ([-]?\d+)')
        return int(pattern.findall(s)[0])

    @staticmethod
    def float(s, field):
        pattern = re.compile(f'{field} = ([-]?\d*\.\d+|\d+)')
        return float(pattern.findall(s)[0])

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize logs')
    parser.add_argument('file', type=str, metavar='F',
                        help='log file to visualize')
    return parser.parse_args()

def parse_data(filename):
    data = {
        'G_MLE': {
            'x': {
                'name': 'epoch',
                'values': []
            },
            'ys': [
                {
                    'name': 'nll',
                    'values': []
                }
            ]
        },
        'G_PG': {
            'x': {
                'name': 'iter',
                'values': []
            },
            'ys': [
                {
                    'name': 'nll',
                    'values': []
                }
            ]
        },
        'D': {
            'x': {
                'name': 'iter',
                'values': []
            },
            'ys': [
                {
                    'name': 'loss',
                    'values': []
                }, {
                    'name': 'train_acc',
                    'values': []
                }, {
                    'name': 'val_acc',
                    'values': []
                }
            ]
        }
    }

    # Extract data
    with open(filename, 'r') as fin:
        for row in fin:
            row = row.rstrip('\n')

            model_type = Extractor.square_bracket(row)
            if model_type == 'G_MLE':
                epoch = Extractor.int(row, 'epoch')
                nll = Extractor.float(row, 'average_train_NLL')
                data[model_type]['x']['values'] += [epoch]
                data[model_type]['ys'][0]['values'] += [nll]
            elif model_type == 'G_PG':
                it = Extractor.int(row, 'iter')
                nll = Extractor.float(row, 'average_train_NLL')
                data[model_type]['x']['values'] += [it]
                data[model_type]['ys'][0]['values'] += [nll]
            elif model_type == 'D':
                it = Extractor.int(row, 'iter')
                loss = Extractor.float(row, 'average_loss')
                train_acc = min(1, Extractor.float(row, 'train_acc'))
                val_acc = min(1, Extractor.float(row, 'val_acc'))
                data[model_type]['x']['values'] += [it]
                data[model_type]['ys'][0]['values'] += [loss]
                data[model_type]['ys'][1]['values'] += [train_acc]
                data[model_type]['ys'][2]['values'] += [val_acc]

    # Gather data across iters/epochs
    for model_type, values in data.items():
        X = values['x']['values']
        YS = values['ys']

        # Gather
        m = defaultdict()
        for i, x in enumerate(X):
            m.setdefault(x, [[0, 0] for _ in range(len(YS))]) # [sum, num_of_elements]
            for j, y in enumerate(YS):
                m[x][j][0] += y['values'][i]
                m[x][j][1] += 1

        # Clear original data
        data[model_type]['x']['values'] = []
        for j, y in enumerate(YS):
            data[model_type]['ys'][j]['values'] = []

        # Put gathered data
        for x, ys in m.items():
            data[model_type]['x']['values'] += [x]
            for j, y in enumerate(ys):
                data[model_type]['ys'][j]['values'] += [y[0] / y[1]]

    return data

def visualize(data):
    # G_MLE pretraining loss, G_PG training loss, & D training loss
    f, axarr = plt.subplots(len(data), 1)
    f.suptitle('Training Losses')

    for i, (model_type, d) in enumerate(data.items()):
        axarr[i].plot(d['x']['values'], d['ys'][0]['values'])
        axarr[i].set(xlabel=d['x']['name'], ylabel=d['ys'][0]['name'])
        axarr[i].set_title(model_type)
        
        ticklabels = list(map(lambda x: str(x) if x != -1 else 'pretrained', d['x']['values']))
        axarr[i].set_xticks(d['x']['values'])
        axarr[i].set_xticklabels(ticklabels)

    f.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # D training & valid accuracy 
    f, ax = plt.subplots()
    f.suptitle('Discriminator Accuracy')

    d = data['D']

    l1, = ax.plot(d['x']['values'], d['ys'][1]['values'])
    l2, = ax.plot(d['x']['values'], d['ys'][2]['values'])
    ax.set(xlabel=d['x']['name'], ylabel='Accuracy')
    ax.legend((l1, l2), (d['ys'][1]['name'], d['ys'][2]['name']))

    ticklabels = list(map(lambda x: str(x) if x != -1 else 'pretrained', d['x']['values']))
    ax.set_xticks(d['x']['values'])
    ax.set_xticklabels(ticklabels)

    f.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    data = parse_data(args.file)
    visualize(data)
