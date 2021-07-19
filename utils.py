import numpy as np
import torch
from math import sin, cos, pi
import random
import matplotlib.pyplot as plt
import matplotlib.colors
from skimage import io
from PIL import Image
import string


def pad_vector(vectors, max_sentence_len):
    if len(vectors) < max_sentence_len:
        for _ in range(max_sentence_len - len(vectors)):
            vectors.append(np.zeros(300, dtype=np.float32))
    elif len(vectors) > max_sentence_len:
        print("max_sentence_len too small, see utils line 16")
        print(sdfsdfsfd)
    return vectors

# return the caption encoding
def get_caption_vector(text_model, caption_without_punctuation, max_sentence_len, encoding_weight=1):
    words_list = caption_without_punctuation.lower().split()
    vectors = [text_model.get_word_vector(word) for word in words_list]
    return np.array(pad_vector(vectors, max_sentence_len))
    #return text_model.get_sentence_vector(caption.replace('\n', '').lower()) * encoding_weight

# get a batch of noise vectors
def get_noise_tensor(batch_size, noise_size):
    """number x 128 x 1 x 1"""
    noise_tensor = torch.randn((batch_size, noise_size, 1, 1), dtype=torch.float32)
    return noise_tensor

if __name__ == "__main__":
    print(np.zeros(3, dtype=np.float32))