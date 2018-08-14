DEBUG = True 
LIGHT_VER = True # Light version of dataset

END_TOKEN = '<E>'
PAD_TOKEN = '<P>'

CUDA = False
MAX_SEQ_LEN_PADDING = 10
BATCH_SIZE = 32
ROLLOUT_NUM = 3
TEACHER_FORCING_RATIO = 0.6
G_PRETRAIN_EPOCHS = 1 if DEBUG else 50
D_PRETRAIN_STEPS = 1 if DEBUG else 50
D_PRETRAIN_EPOCHS = 1 if DEBUG else 3
G_TRAIN_STEPS = 1 if DEBUG else 1
D_TRAIN_STEPS = 1 if DEBUG else 5
D_TRAIN_EPOCHS = 1 if DEBUG else 3
ADV_TRAIN_ITERS = 3 if DEBUG else 50

G_ED = 32 # embedding dim
G_HD = 32 # hidden dim
D_ED = 64
D_HD = 64

VALID_SET_SIZE_RATIO = 0.2
SAVE_MODEL_ITER = 1

dataset_path = './dataset/quora_duplicate_questions.tsv'

model_params = { 'gan': { 'rn': ROLLOUT_NUM, 'tfr': TEACHER_FORCING_RATIO, 'bs': BATCH_SIZE, 'pad': MAX_SEQ_LEN_PADDING },
                 'G': { 'ed': G_ED, 'hd': G_HD },
                 'D': { 'ed': D_ED, 'hd': D_HD }}
training_params = { 'gan': { 'iter': ADV_TRAIN_ITERS },
                    'G': { 'st': G_TRAIN_STEPS, 'ep': 1 },
                    'D': { 'st': D_TRAIN_STEPS, 'ep': D_TRAIN_EPOCHS }}
pretrain_params = { 'G': { 'st': 1, 'ep': G_PRETRAIN_EPOCHS },
                    'D': { 'st': D_PRETRAIN_STEPS, 'ep': D_PRETRAIN_EPOCHS }}
