# SeqGAN Paraphrase Generation

# Dataset

- [Quora](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)

# References

- [LantaoYu/SeqGAN](https://github.com/LantaoYu/SeqGAN)
- [suragnair/seqGAN](https://github.com/suragnair/seqGAN)

# Usage

```bash
# Install dependencies
$ conda create -n <env-name> --file requirements.txt # if using Anaconda to create virtual env
$ source activate <env-name>                         # enter virtual env
## OR other methods, e.g.
$ pip3 install -r requirements.txt

# Parse pretrained word embeddings (currently only GloVe is supported)
$ python3 tools/parse_emb.py <word-embedding-file> <output-vector-file> <output-info-file>
## Vector file: *.npy, which keeps a numpy array
## Info file: *.pkl, which keeps the word mapping information

# Run GAN training
$ python3 -m src.train
```

# Project structure

```bash
.
├── dataset
│   ├── pretrained_word_embeddings    # not included by git, download word embedding dataset and see Usage to create parsed files
│   │   ├── glove_50_info.pkl         # word mapping information
│   │   ├── glove_50.npy              # pretrained word vectors
│   │   └── original                  # raw vectors, downloaded from https://nlp.stanford.edu/projects/glove/
│   │       ├── glove.twitter.27B.50d.txt
│   │       └── ...
│   └── quora_duplicate_questions.tsv # downloaded from https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs
├── environment.yml                   # conda env details
├── log                               # storing logs named by timestamp and hyperparameters
├── model                             # storing trained models
├── README.md
├── requirements.txt                  # dependency details
├── src
│   ├── discriminator.py              # discriminator model
│   ├── generator.py                  # generator model
│   ├── __init__.py
│   ├── train.py                      # training script
│   └── utils
│       ├── dataloader.py             # load dataset, parse into positive samples & condition pairs
│       ├── helpers.py                # helper methods
│       ├── hyper_params.py           # hyperparameters e.g. hidden dimension, training epoch, etc.
│       ├── pathbuilder.py            # build model save paths given model parameters; auto-detect trained models and resume training
│       ├── static_params.py          # static project settings e.g. debug mode, dataset path, etc.
│       └── word_embeddings.py        # word embedding helper class, load or initialize word embeddings
└── tools
    ├── __init__.py
    ├── paraphrase_generation.py      # interactive script, generate paraphrase given a sentence
    ├── parse_emb.py                  # parse word embedding vectors and mappings from raw file
    └── visualize_log.py              # visualize loss, acc, etc. information in a given log file`
