import numpy as np
import torch
from math import sin, cos, pi
import random
import matplotlib.pyplot as plt
import matplotlib.colors
from skimage import io
from PIL import Image



# return the caption encoding
def get_caption_vector(text_model, caption, encoding_weight):
    return text_model.get_sentence_vector(caption.replace('\n', '').lower()) * encoding_weight

# get a batch of noise vectors
def get_noise_tensor(batch_size, noise_size):
    """number x 128 x 1 x 1"""
    noise_tensor = torch.randn((batch_size, noise_size, 1, 1), dtype=torch.float32)
    return noise_tensor

