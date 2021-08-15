from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict
import os

project_dir = os.path.dirname(__file__) # /media/remote_home/chang/z_master-thesis
my_dir = os.path.dirname(project_dir) # /media/remote_home/chang

__C = edict()
cfg = __C

__C.CUDA = True
__C.USE_TENSORBOARD = True

__C.START_FROM_EPOCH = 0#200
__C.END_IN_EPOCH = 1200#1200
__C.CHKPT_PATH = project_dir + '/models/checkpoints'

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

# training
__C.N_train_D_1_train_G = 10 # train discriminator k times before training generator
__C.BATCH_SIZE = 128
__C.LEARNING_RATE_G = 0.00001
__C.LEARNING_RATE_D = 0.00001
__C.NOISE_SIZE = 150
__C.COMPRESS_SIZE = 128
__C.WORKERS = 8
__C.JOINT_NUM = 24
__C.MAX_SENTENCE_LEN = __C.JOINT_NUM

__C.D_WORD_VEC = 150

__C.SCORE_WRONG_WEIGHT_D = 2 # SCORE_RIGHT_WEIGHT_D
__C.SCORE_FAKE_WEIGHT_D = 1 # SCORE_RIGHT_WEIGHT_D
__C.SCORE_RIGHT_WEIGHT_D = __C.SCORE_WRONG_WEIGHT_D + __C.SCORE_FAKE_WEIGHT_D
__C.PENALTY_WEIGHT_WRONG = 10
__C.PENALTY_WEIGHT_FAKE = 10

# Model g
__C.D_MODEL_LIST_G = [300,128,64,32,16,8,3] 
__C.N_LAYERS_G = len(__C.D_MODEL_LIST_G)-1
__C.N_HEAD_LIST_G = [8,8,4,4,4,4]
__C.D_K_LIST_G = [64,32,16,8,4,4] 
__C.D_V_LIST_G = __C.D_K_LIST_G
__C.DROPOUT_G = 0.1
__C.N_POSITION_G = 200
__C.SCALE_EMB_G = False
__C.D_INNER_SCALE_G = 2


# Model d
__C.D_MODEL_LIST_D = [300,128,64,32,16,8,1] 
__C.N_LAYERS_D = len(__C.D_MODEL_LIST_D)-1
__C.N_HEAD_LIST_D = [8,8,4,4,4,4,2]
__C.D_K_LIST_D = [64,32,16,8,4,4] 
__C.D_V_LIST_D = __C.D_K_LIST_D
__C.DROPOUT_D = 0.1
__C.N_POSITION_D = 200
__C.SCALE_EMB_D = False
__C.D_INNER_SCALE_D = 2
__C.FAKE_WRONG_DIFF_WEIGHT = 1

# OPtimizer
# ADAM solver
__C.BETA_1 = 0.5
__C.BETA_2 = 0.999
__C.N_WARMUP_STEPS_G = 2000
__C.N_WARMUP_STEPS_D = 3000

# numbers of channels of the convolutions
__C.CONVOLUTION_CHANNEL_G = [256, 128, 64, 32]
__C.CONVOLUTION_CHANNEL_D = [32, 64, 128, 256]

__C.GENERATOR_PATH = project_dir + '/models/generator'
__C.DISCRIMINATOR_PATH = project_dir + '/models/discriminator'
__C.OUTPUT_DIR = project_dir + '/output'

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
