import torch


DEBUG = True
LIGHT_VER = True # Light version of dataset
NO_SAVE = False

END_TOKEN = '<E>'
PAD_TOKEN = '<P>'

CUDA = torch.cuda.is_available()

VALID_SET_SIZE_RATIO = 0.2
SAVE_MODEL_ITER = 1

dataset_path = './dataset/quora_duplicate_questions.tsv'
