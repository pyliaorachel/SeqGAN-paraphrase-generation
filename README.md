# SeqGAN Paraphrase Generation

# Dataset

- [Quora](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)

# References

- [LantaoYu/SeqGAN](https://github.com/LantaoYu/SeqGAN)
- [suragnair/seqGAN](https://github.com/suragnair/seqGAN)

# Usage

## Preparation

###### 1: Install dependencies

1. Anaconda, default env name: `seqgan` (can change this in `environment.yml`)
    1. Create virtual environment: `$ conda env create -f environment.yml`
    2. Enter virtual env: `$ source activate <env-name>`
2. Other methods
    1. `$ pip3 install -r requirements.txt`

###### 2: Activate nltk

`nltk.word_tokenize()` method needs `punkt` package.

```bash
$ python3
>>> import nltk
>>> nltk.download('punkt')
```

###### 3: Install nlg-eval (for evaluation only)

From instructions here: https://github.com/Maluuba/nlg-eval

```bash
$ git clone https://github.com/Maluuba/nlg-eval.git
$ cd nlg-eval
$ pip3 install -e .
$ nlg-eval --setup
```

###### 4: Download pretrained word embeddings

- Download pretrained word embeddings
    - [GloVe](https://nlp.stanford.edu/projects/glove/)
    - [word2vec](https://code.google.com/archive/p/word2vec/)
    - [fastText](https://github.com/icoxfog417/fastTextJapaneseTutorial)
    - [ELMo](https://allennlp.org/elmo)
- Extract into plain text files & put under `dataset/pretrained_word_embeddings/original/`

###### 5: Parse pretrained word embeddings
    
```bash
$ python3 tools/parse_emb.py <word-embedding-file> <output-vector-file> <output-info-file>
## Vector file: <output-dir>/<word-emb>_<emb-dim>.npy, which keeps a numpy array
## Info file: <output-dir>/<word-emb>_<emb-dim>_info.pkl, which keeps the word mapping information
```

\* _Currently only GloVe is tested; if you want to substitute it with others, simply make sure the pretrained embedding file format is consistent with that of GloVe's, and change the `pretrained_emb` parameter in `src/utils/hyper_params.py` to your choice._

###### Summary

```bash
$ conda env create -f environment.yml
$ source activate seqgan

$ python3
>>> import nltk
>>> nltk.download('punkt')

$ git clone https://github.com/Maluuba/nlg-eval.git
$ cd nlg-eval
$ pip3 install -e .
$ nlg-eval --setup

# Download pretrained word embeddings & extract

$ python3 tools/parse_emb.py dataset/pretrained_word_embeddings/original/glove.twitter.27B.50d.txt dataset/pretrained_word_embeddings/glove_50.npy dataset/pretrained_word_embeddings/glove_50_info.pkl
```

## Train Model

```bash
$ python3 -m src.train
# Output model files: model/<dataset-info>/<model-params>/<gen/dis>.trc
# Output pretrained model files: model/<dataset-info>/<model-params>/pretrain/<pretrained-model-params>/<gen/dis>.trc
# Output log file: log/<timestamp>_<dataset-info>_<model-params>_<pretrained-model-params>.log
```

- Hyperparameters: `src/utils/hyper_params.py`
  - Batch size, rollout number, training epochs/steps, etc.
- Project parameters: `src/utils/static_params.py`
  - Debug mode (run fewer iterations), light mode (load smaller dataset), save mode (save model or not)
  - Train/test/validation set sizes
  
###### Continue training

If you just trained for some iterations and would like to pick up from that point, simply change the `ADV_TRAIN_ITERS` param in `src/utils/hyper_params.py` to the number of __additional__ training iterations, leave all other parameters intact, and rerun. Our smart pathbuilder will detect the existing model with the same parameters, load that model, and resume training.

## Evaluate Model

Paraphrases will be generated, and the BLEU-2 and METEOR evaluation metrics will be calculated. The model path below should be the directory path of the pretrained model and end in slash. Our pathbuilder tool will parse everything with this.

```bash
$ python3 -m tools.evaluate model/<model-params>/pretrain/<pretrained-model-params>/
# Output results to output/<model-params>/pretrain/<pretrained-model-params>/results.tsv
```

- Options
    - Only evaluate pretrained model: `--pretrained`
    - Don't include evaluation metric scores: `--no-score`
        - Output file name: `results_raw.tsv`
    - Evaluate on training set instead of test set: `--mode train`
    
Output file format:

```
original (cond) sample (pos)    generated (neg) BLEU    METEOR
This is the first sentence. This sentence is the first one. This one is generated.  1.005427487071676e-08   0.1839080459770115
```

## Other Tools

- Visualize log file with loss & accuracy information
    - `$ python3 tools/visualize_log.py <path-to-log-file> plots/`
    - Output `*_acc.png` and `*_loss.png`
- Interactive paraphrase generation
    - `$ python3 -m tools.paraphrase_generation <path-to-pretrained-model-dir>`
- Find paraphrases with top-n scores
    - `$ python3 tools/find_good_ex.py <path-to-result-file> <output-filename> -n <n> --metric <metric-to-compare>`

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
├── plots                             # plots of log files, including accuracy & loss trends
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
    ├── evaluate.py                   # evaluation script, output generated paraphrases and evaluation scores
    ├── find_good_ex.py               # find generated samples with the best n scores
    ├── paraphrase_generation.py      # interactive script, generate paraphrase given a sentence
    ├── parse_emb.py                  # parse word embedding vectors and mappings from raw file
    └── visualize_log.py              # visualize loss, acc, etc. information in a given log file`
