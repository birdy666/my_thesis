import numpy as np
import torch
from math import sin, cos, pi
import random
import string
import time

def get_caption_mask(vectors, max_sentence_len):
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
    mask = np.array(get_caption_mask(vectors, max_sentence_len))
    return np.array(pad_vector(vectors, max_sentence_len, d_word_vec)), mask

"""TODO 24is hyperparameter"""
# get a batch of noise vectors
def get_noise_tensor(batch_size, noise_size):
    """batch_size x 24 x noise_size"""
    noise_tensor = torch.randn((batch_size, 1, noise_size), dtype=torch.float32)
    return noise_tensor

def print_performances(header, start_time, loss_g, loss_d, lr_g, lr_d, e):
    print('  - {header:12} epoch {e}, loss_g: {loss_g: 8.5f}, loss_d: {loss_d:8.5f} %, lr_g: {lr_g:8.5f}, lr_d: {lr_d:8.5f}, '\
            'elapse: {elapse:3.3f} min'.format(
                e = e, header=f"({header})", loss_g=loss_g,
                loss_d=loss_d, elapse=(time.time()-start_time)/60, lr_g=lr_g, lr_d=lr_d))

def save_models(cfg, e, net_g, net_d, n_steps_g, n_steps_d, chkpt_path,  save_mode='all'):
    checkpoint = {'epoch': e, 'model_g': net_g.state_dict(), 'model_d': net_d.state_dict(),
                    'n_steps_g': n_steps_g, 'n_steps_d':n_steps_d, 'cfg':cfg}

    if save_mode == 'all':
        torch.save(checkpoint, chkpt_path + "/epoch_" + str(e) + ".chkpt")
    elif save_mode == 'best':
        pass
        """model_name = 'model.chkpt'
        if valid_loss <= min(valid_losses):
            torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
            print('    - [Info] The checkpoint file has been updated.')"""