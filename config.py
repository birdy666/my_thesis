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
__C.USE_TENSORBOARD = False

__C.START_FROM_EPOCH = 0#200
__C.END_IN_EPOCH = 1#1200

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
__C.SENTENCE_VECTOR_SIZE = 300

# smpl
__C.SMPL_MODEL_PATH = my_dir + "/datasets/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"

# training
__C.N_train_D_1_train_G = 5 # train discriminator k times before training generator
__C.BATCH_SIZE = 128
__C.LEARNING_RATE_G = 0.0004
__C.LEARNING_RATE_D = 0.0004
__C.NOISE_SIZE = 128
__C.COMPRESS_SIZE = 128
__C.WORKERS = 8
__C.JOINT_NUM = 24
__C.MAX_SENTENCE_LEN = 24
# Model
__C.N_LAYERS = 3
__C.D_INNER_SCALE = 2
__C.D_WORD_VEC = 300
__C.D_MODEL_LIST = [600,75,15] 
__C.N_HEAD_LIST = [8,5,5]
__C.D_K_LIST = [75,15,3] # 這裡我實在不知道d_k和d_model之間到底要不要有N_head的倍數關西?
__C.D_V_LIST = __C.D_K_LIST
__C.DROPOUT = 0.1
__C.N_POSITION = 200
__C.SCALE_EMB = False

# ADAM solver
__C.BETA_1 = 0.0
__C.BETA_2 = 0.9
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
