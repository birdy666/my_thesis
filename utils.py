import numpy as np
import torch
from math import sin, cos, pi
import random
import string

def get_input_mask(vectors, max_sentence_len):
    mask = []
    for _ in range(len(vectors)):
        mask.append(1)
    for _ in range(len(vectors), max_sentence_len):
        mask.append(0)
    return mask

def pad_vector(vectors, max_sentence_len, d_word_vec):
    if len(vectors) < max_sentence_len:
        for _ in range(max_sentence_len - len(vectors)):
            vectors.append(np.zeros(d_word_vec, dtype=np.float32))
    elif len(vectors) > max_sentence_len:
        print("max_sentence_len too small, see utils line 16")
        print(sdfsdfsfd)
    return vectors

# return the caption encoding
def get_caption_vector(text_model, caption_without_punctuation, max_sentence_len, d_word_vec, encoding_weight=1):
    words_list = caption_without_punctuation.lower().split()
    vectors = [text_model.get_word_vector(word) for word in words_list]
    mask = np.array(get_input_mask(vectors, max_sentence_len))
    return np.array(pad_vector(vectors, max_sentence_len, d_word_vec)), mask
    #return text_model.get_sentence_vector(caption.replace('\n', '').lower()) * encoding_weight

"""TODO 24is hyperparameter"""
# get a batch of noise vectors
def get_noise_tensor(batch_size,  noise_size):
    """batch_size x 24 x 300"""
    noise_tensor = torch.randn((batch_size, 24, noise_size), dtype=torch.float32)
    return noise_tensor

if __name__ == "__main__":
    print(random.choice(list(range(2))))