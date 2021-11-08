from __future__ import division
from __future__ import print_function
import math
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
import os
from torch.utils.tensorboard import SummaryWriter

project_dir = os.path.dirname(__file__) # /media/remote_home/chang/z_master-thesis
my_dir = os.path.dirname(project_dir) # /media/remote_home/chang

__C = edict()
cfg = __C

__C.CUDA = True
__C.USE_TENSORBOARD = True

__C.START_FROM_EPOCH = 0#200
__C.END_IN_EPOCH = 20000#1200
__C.CHKPT_PATH = project_dir + '/models/checkpoints'
__C.TB_DIR = project_dir + '/output/tensorboard'
__C.SAVE_MODEL_ITR = 500

# training
__C.N_train_D_1_train_G = 2# train discriminator k times before training generator
__C.BATCH_SIZE = 128*1
__C.LEARNING_RATE_G = 2e-5
__C.LEARNING_RATE_D = 2e-5
__C.WEIGHT_DECAY_G = 0
__C.WEIGHT_DECAY_D = 0.01

__C.COMPRESS_SIZE = 128
__C.WORKERS = 8
__C.JOINT_NUM = 24
__C.MAX_SENTENCE_LEN = __C.JOINT_NUM

__C.D_WORD_VEC = 72
__C.NOISE_SIZE = __C.D_WORD_VEC

w_d = 1
f_d = 1
r_d = 2


f_g = 8
i_g = 2

__C.SCORE_WRONG_WEIGHT_D = w_d/(w_d+f_d+r_d)
__C.SCORE_FAKE_WEIGHT_D = f_d/(w_d+f_d+r_d)
__C.SCORE_RIGHT_WEIGHT_D = r_d/(w_d+f_d+r_d)
__C.PENALTY_WEIGHT_WRONG = 1
__C.PENALTY_WEIGHT_FAKE = 1
__C.SCORE_FAKE_WEIGHT_G = f_g/(f_g+i_g)
__C.SCORE_INTERPOLATE_WEIGHT_G = i_g/(f_g+i_g)

# Model g
__C.ENC_PARAM_G = edict({'n_layers':8, 'd_model':__C.D_WORD_VEC, 'd_inner_scale':4, 'n_head':4, 
                'd_k':32, 'd_v':32, 'dropout':0.1, 'scale_emb':False})
__C.DEC_PARAM_G = edict({'n_layers':2, 'd_model':__C.D_WORD_VEC, 'd_inner_scale':4, 'n_head':4, 
                'd_k':32, 'd_v':32, 'dropout':0.2, 'scale_emb':False})
__C.FC_LIST_G = []
__C.DROPOUT_G = 0.2


# Model d
__C.ENC_PARAM_D = edict({'n_layers':6, 'd_model':__C.D_WORD_VEC, 'd_inner_scale':4, 'n_head':4, 
                'd_k':32, 'd_v':32, 'dropout':0.1, 'scale_emb':False})
__C.DEC_PARAM_D = edict({'n_layers':6, 'd_model':__C.D_WORD_VEC, 'd_inner_scale':4, 'n_head':4, 
                'd_k':32, 'd_v':32, 'dropout':0.2, 'scale_emb':False})
__C.FC_LIST_D = []
__C.DROPOUT_D = 0.2

# OPtimizer
# ADAM solver
__C.BETA_1 = 0.9
__C.BETA_2 = 0.999
__C.N_WARMUP_STEPS_G = 2000
__C.N_WARMUP_STEPS_D = 3000

# models/output
__C.GENERATOR_PATH = project_dir + '/models/generator'
__C.DISCRIMINATOR_PATH = project_dir + '/models/discriminator'
__C.OUTPUT_DIR = project_dir + '/output'

# coco
__C.DATA_DIR = my_dir + "/datasets/coco"
__C.DATASET_NAME = 'coco'
__C.COCO_CAPTION_TRAIN = __C.DATA_DIR + '/annotations/captions_train2014.json'
__C.COCO_CAPTION_VAL = __C.DATA_DIR + '/annotations/captions_val2014.json'
__C.COCO_keypoints_TRAIN = __C.DATA_DIR + '/annotations/person_keypoints_train2014.json'
__C.COCO_VAL_PORTION = 0.1

# EFT
__C.EFT_FIT_DIR = my_dir + "/eft/eft_fit"
__C.EFT_FIT_PATH = my_dir + "/eft/eft_fit/COCO2014-All-ver01.json"
__C.EFT_FIT_WITH_CAPTION_PATH = my_dir + '/eft/eft_fit/COCO2014-All-ver01_with_caption.json'

# text model
__C.TEXT_MODEL_PATH = my_dir + '/fastText/wiki.en.bin'
__C.ENCODING_WEIGHT = 30

# smpl
__C.SMPL_MODEL_PATH = my_dir + "/datasets/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
