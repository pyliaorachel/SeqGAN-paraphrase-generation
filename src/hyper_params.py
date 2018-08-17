from .static_params import *


MAX_SEQ_LEN_PADDING = 10
BATCH_SIZE = 16
ROLLOUT_NUM = 3
TEACHER_FORCING_RATIO = 0.6
G_PRETRAIN_EPOCHS = 1 if DEBUG else 50
D_PRETRAIN_STEPS = 1 if DEBUG else 3
D_PRETRAIN_EPOCHS = 1 if DEBUG else 2
G_TRAIN_STEPS = 1 if DEBUG else 1
D_TRAIN_STEPS = 1 if DEBUG else 3
D_TRAIN_EPOCHS = 1 if DEBUG else 2
ADV_TRAIN_ITERS = 3 if DEBUG else 20

G_ED = 32 # embedding dim
G_HD = 32 # hidden dim
D_ED = 64
D_HD = 64

model_params = { 'gan': { 'rn': ROLLOUT_NUM, 'tfr': TEACHER_FORCING_RATIO, 'bs': BATCH_SIZE, 'pad': MAX_SEQ_LEN_PADDING },
                 'G': { 'ed': G_ED, 'hd': G_HD },
                 'D': { 'ed': D_ED, 'hd': D_HD }}
training_params = { 'gan': { 'iter': ADV_TRAIN_ITERS },
                    'G': { 'st': G_TRAIN_STEPS, 'ep': 1 },
                    'D': { 'st': D_TRAIN_STEPS, 'ep': D_TRAIN_EPOCHS }}
pretrain_params = { 'G': { 'st': 1, 'ep': G_PRETRAIN_EPOCHS },
                    'D': { 'st': D_PRETRAIN_STEPS, 'ep': D_PRETRAIN_EPOCHS }}