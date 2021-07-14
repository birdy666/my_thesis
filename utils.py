import numpy as np
import torch
from math import sin, cos, pi
import random
import matplotlib.pyplot as plt
import matplotlib.colors
from skimage import io
from PIL import Image

from hyperparams import *
from path import *


# return the caption encoding
def get_caption_vector(cfg, text_model, caption):
    return text_model.get_sentence_vector(caption.replace('\n', '').lower()) * cfg.ENCODING_WEIGHT


# get a batch of noise vectors
def get_noise_tensor(batch_size):
    """number x 128 x 1 x 1"""
    noise_tensor = torch.randn((batch_size, noise_size, 1, 1), dtype=torch.float32)
    return noise_tensor

