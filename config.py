from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict

chang = "/media/remote_home/chang"

__C = edict()
cfg = __C

__C.CUDA = True

# coco
__C.DATA_DIR = chang + "/datasets/coco"
__C.DATASET_NAME = 'coco'
__C.COCO_CAPTION_TRAIN = __C.DATA_DIR + '/annotations/captions_train2014.json'
__C.COCO_CAPTION_VAL = __C.DATA_DIR + '/annotations/captions_val2014.json'
__C.COCO_VAL_PORTION = 0.1

# EFT
__C.EFT_FIT_DIR = chang + "/eft/eft_fit"
__C.EFT_FIT_PATH = chang + "/eft/eft_fit/COCO2014-All-ver01.json"
__C.EFT_FIT_WITH_CAPTION_PATH = chang + '/eft/eft_fit/COCO2014-All-ver01_with_caption.json'

# text model
__C.TEXT_MODEL_PATH = chang + '/fastText/wiki.en.bin'
__C.ENCODING_WEIGHT = 30
__C.SENTENCE_VECTOR_SIZE = 300

# smpl
__C.SMPL_MODEL_PATH = chang + "/datasets/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"

# training
__C.BATCH_SIZE = 128
__C.LEARNING_RATE_G = 0.0004
__C.LEARNING_RATE_D = 0.0004
__C.START_FROM_EPOCH = 0#200
__C.END_IN_EPOCH = 1#1200
__C.NOISE_SIZE = 128
__C.COMPRESS_SIZE = 128
__C.WORKERS = 8
# ADAM solver
__C.BETA_1 = 0.0
__C.BETA_2 = 0.9
# numbers of channels of the convolutions
__C.CONVOLUTION_CHANNEL_G = [256, 128, 64, 32]
__C.CONVOLUTION_CHANNEL_D = [32, 64, 128, 256]

__C.GENERATOR_PATH = chang + '/z_master-thesis/models/generator'
__C.DISCRIMINATOR_PATH = chang + '/z_master-thesis/models/discriminator'


"""# Dataset name: flowers, birds
__C.DATASET_NAME = 'coco'
__C.CONFIG_NAME = ''
__C.DATA_DIR = chang + "/datasets/coco"
__C.GPU_ID = 0
__C.CUDA = True
__C.WORKERS = 6

__C.RNN_TYPE = 'LSTM'   # 'GRU'
__C.B_VALIDATION = False
__C.loss = 'hinge'
__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64


# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = ''
__C.TRAIN.NET_G = ''
__C.TRAIN.B_NET_D = True
__C.TRAIN.NF = 32
__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0


# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.R_NUM = 2
__C.GAN.B_ATTENTION = True
__C.GAN.B_DCGAN = True


__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 10
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 18
__C.TEXT.DAMSM_NAME = '../DAMSMencoders/coco/text_encoder200.pth'"""

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
