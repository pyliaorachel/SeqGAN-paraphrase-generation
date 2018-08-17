import torch


DEBUG = False
LIGHT_VER = False # Light version of dataset
NO_SAVE = False

END_TOKEN = '<E>'
PAD_TOKEN = '<P>'

CUDA = torch.cuda.is_available()

VALID_SET_SIZE_RATIO = 0.1

dataset_path = './dataset/quora_duplicate_questions.tsv'