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


"""
Plot something
"""

# plot a pose
def plot_pose(x, y, v, skeleton, caption=None):
    # visible keypoints
    keypoint_show = np.arange(total_keypoints)[v > 0]

    # visible skeleton
    skeleton = skeleton[0:17]
    x_skeleton = x[skeleton]
    y_skeleton = y[skeleton]
    skeleton_show = [i for i in range(len(skeleton)) if (v[skeleton[i]] > 0).all()]

    # plot keypoints and skeleton
    plt.imshow(np.ones((64, 64, 3), 'float32'))
    [plt.plot(x_skeleton[i], y_skeleton[i], c=skeleton_colors[i], linewidth=5) for i in skeleton_show]
    [plt.plot(x[i], y[i], 'o', c=keypoint_colors[i], markersize=10, markeredgecolor='k', markeredgewidth=1) for i in
     keypoint_show]
    plt.title('pose')
    plt.xlabel(caption)


#
def plot_generative_samples_from_noise(fixed_fake, fixed_real_array, fixed_caption, fixed_w, fixed_h, multi, skeleton, start_from_epoch):
    fixed_fake = np.array(fixed_fake.tolist()) * 0.5 + 0.5
    f = plt.figure(figsize=(19.2, 12))
    for sample in range(fixed_w):
        plt.subplot(fixed_h + 1, fixed_w, sample + 1)
        plot_heatmap(fixed_real_array[sample], skeleton=(None if multi else skeleton))
        plt.title(fixed_caption[sample][0:30] + '\n' + fixed_caption[sample][30:])
        plt.xlabel('(real)')
        plt.xticks([])
        plt.yticks([])
    for sample in range(fixed_w*fixed_h):
        plt.subplot(fixed_h + 1, fixed_w, fixed_w + sample + 1)
        plot_heatmap(fixed_fake[sample], skeleton=(None if multi else skeleton))
        plt.title(None)
        plt.xlabel('(fake)')
        plt.xticks([])
        plt.yticks([])
    plt.savefig(figures_path + 'fixed_noise_samples_' + f'{start_from_epoch:05d}' + '_new.png')
    plt.close()


