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
def get_caption_vector(text_model, caption):
    return text_model.get_sentence_vector(caption.replace('\n', '').lower()) * encoding_weight


# get a batch of noise vectors
def get_noise_tensor(number):
    """number x 128 x 1 x 1"""
    noise_tensor = torch.randn((number, noise_size, 1, 1), dtype=torch.float32)
    return noise_tensor


# get max index of a pose
def coordinates_to_max_index(x, y, v):
    max_index = np.array([x, y]).transpose()

    # set the invisible keypoints to the middle
    max_index[v < 1] = [heatmap_size / 2, heatmap_size / 2]
    return max_index


# find the nearest neighbor distance of a heatmap in a list of heatmaps
def nearest_neighbor(heatmap_max_index, heatmap_max_index_list):
    distance = heatmap_distance(heatmap_max_index, heatmap_max_index_list[0])

    # find nearest neighbor
    for heatmap_max_index2 in heatmap_max_index_list[1:]:
        new_distance = heatmap_distance(heatmap_max_index, heatmap_max_index2)
        if new_distance < distance:
            distance = new_distance
    return distance


# find the nearest neighbor of a heatmap in a list of heatmaps
def nearest_neighbor_index(heatmap_max_index, heatmap_max_index_list):
    distance = heatmap_distance(heatmap_max_index, heatmap_max_index_list[0])
    index = 0

    # find nearest neighbor
    for i in range(len(heatmap_max_index_list)):
        new_distance = heatmap_distance(heatmap_max_index, heatmap_max_index_list[i])
        if new_distance < distance:
            distance = new_distance
            index = i
    return index


# calculate the mean distance of a heatmap to a list of heatmaps
def mean_distance(heatmap_max_index, heatmap_max_index_list):
    distance = []

    # calculate distances
    for heatmap_max_index2 in heatmap_max_index_list:
        distance.append(heatmap_distance(heatmap_max_index, heatmap_max_index2))

    return np.mean(distance)


# calculate the mean distance of a vector to a list of vectors
def mean_vector_distance(vector, vector_list):
    distance = []

    # calculate distances
    for vector2 in vector_list:
        distance.append(np.sqrt(np.sum((vector - vector2) ** 2)))

    return np.mean(distance)


# find the nearest neighbor of a vector in a list of vectors
def vector_nearest_neighbor_index(vector, vector_list):
    distance = np.sqrt(np.sum((vector - vector_list[0]) ** 2))
    index = 0

    # find nearest neighbor
    for i in range(len(vector_list)):
        new_distance = np.sqrt(np.sum((vector - vector_list[i]) ** 2))
        if new_distance < distance:
            distance = new_distance
            index = i
    return index


# calculate the one-nearest-neighbor accuracy
def one_nearest_neighbor(heatmap_max_index_list, heatmap_max_index_list2):
    size = len(heatmap_max_index_list)

    # number of correct classifications
    count = 0
    for i in range(size):
        # a heatmap from the first list
        if nearest_neighbor(heatmap_max_index_list[i],
                            heatmap_max_index_list[0:i] + heatmap_max_index_list[i + 1:]) < nearest_neighbor(
            heatmap_max_index_list[i], heatmap_max_index_list2):
            count = count + 1

        # a heatmap from the second list
        if nearest_neighbor(heatmap_max_index_list2[i],
                            heatmap_max_index_list2[0:i] + heatmap_max_index_list2[i + 1:]) < nearest_neighbor(
            heatmap_max_index_list2[i], heatmap_max_index_list):
            count = count + 1

    # accuracy
    return count / size / 2


# return parameters for an augmentation
def get_augment_parameters(flip, scale, rotate, translate):
    # random flip, rotation, scaling, translation
    f = random.random() < flip
    r = random.uniform(-rotate, rotate) * pi / 180
    s = random.uniform(scale, 1 / scale)
    tx = random.uniform(-translate, translate)
    ty = random.uniform(-translate, translate)
    return f, r, s, tx, ty


# return coordinates of keypoints in the heatmap and visibility
def get_coordinates(keypoint_ann, full_image=False):
    """
    (x0, y0): bbox左上角的座標
    (x, y): 特徵點的座標，必定都比x0, y0大，因為這個座標會被包在bbox中
    """
    x0, y0, w, h = tuple(keypoint_ann.get('bbox'))
    if full_image:
        x0 = 0
        y0 = 0
    # keypoints location (x, y) and visibility (v, 0 invisible, 1 visible)
    x = np.array(keypoint_ann.get('keypoints')[0::3])
    y = np.array(keypoint_ann.get('keypoints')[1::3])
    v = np.array(keypoint_ann.get('keypoints')[2::3]).clip(0, 1)

    # calculate the scaling
    if h > w:
        """
        由於bbox可能不是正方形，所以他想把它變成正方形，並取長的邊當作正方形的邊
        這裡先看y，(y - y0) / h就是y座標在正方形bbox上的比例位置，乘上 heatmap_size 就是在heatmap上的位置
        x的話也先用一樣的方式校正，但因為正方形bbox在w方向有h-w的長度是後來加上去的這個長度放到heatmap就是heatmap_size * (1 - w/h)
        為甚麼最後除以二， 因為想像把這多出來的h-w切一半分別加到bbox的左右邊，這樣才不會太偏左
        """
        x_heatmap = (x - x0) / h * heatmap_size + (heatmap_size * (1 - w/h)) / 2
        y_heatmap = (y - y0) / h * heatmap_size
    else:
        x_heatmap = (x - x0) / w * heatmap_size
        y_heatmap = (y - y0) / w * heatmap_size + (heatmap_size * (1 - h/w)) / 2

    # set invisible keypoint coordinates as (0,0)
    x_heatmap[v < 1] = 0
    y_heatmap[v < 1] = 0

    return x_heatmap, y_heatmap, v


# return coordinates of keypoints in the heatmap and visibility with augmentation
def get_augmented_coordinates(keypoint_ann):
    # coordinates and visibility before augmentation
    x, y, v = get_coordinates(keypoint_ann)

    # random flip, rotation, scaling, translation
    f, r, s, tx, ty = get_augment_parameters(flip, scale, rotate, translate)
    x, y, v = augment_heatmap(x, y, v, heatmap_size / 2, f, r, s, tx, ty)

    # set invisible keypoint coordinates as (0,0)
    x[v < 1] = 0
    y[v < 1] = 0

    # concatenate the coordinates and visibility
    return np.concatenate([x, y, v])


# seperate coordinates and visibility from an array
def result_to_coordinates(result):
    x = result[0:total_keypoints]
    y = result[total_keypoints:2 * total_keypoints]
    v = (np.sign(result[2 * total_keypoints:3 * total_keypoints] - v_threshold) + 1) / 2
    return x, y, v

"""
Heatmap
"""

# distance between two heatmaps: the sum of the distances between maximum points of all 17 keypoint heatmaps
def heatmap_distance(heatmap_max_index, heatmap_max_index2):
    return sum(np.sqrt(np.sum((heatmap_max_index - heatmap_max_index2) ** 2, axis=1)))


# get max index of a heatmap
def heatmap_to_max_index(heatmap):
    max_index = np.array([np.unravel_index(np.argmax(h), h.shape) for h in heatmap])

    # set the index of heatmap below threshold to the middle
    for i in range(len(heatmap)):
        if heatmap[i][tuple(max_index[i])] < heatmap_threshold:
            max_index[i][:] = heatmap_size / 2
    return max_index


# return ground truth heatmap of a training sample (fixed-sized square-shaped, can be augmented)
def get_heatmap(keypoint_ann, augment=True):
    # x-y grids
    x_grid = np.repeat(np.array([range(heatmap_size)]), heatmap_size, axis=0)
    y_grid = np.repeat(np.array([range(heatmap_size)]).transpose(), heatmap_size, axis=1)
    empty = np.zeros([heatmap_size, heatmap_size], dtype='float32')

    # heatmap dimension is (number of keypoints)*(heatmap size)*(heatmap size)
    heatmap = np.empty((total_keypoints, heatmap_size, heatmap_size), dtype='float32')

    # keypoints location (x, y) and visibility (v)
    x, y, v = get_coordinates(keypoint_ann)

    # do heatmap augmentation
    if augment:
        # random flip, rotation, scaling, translation
        f, r, s, tx, ty = get_augment_parameters(flip, scale, rotate, translate)
        x, y, v = augment_heatmap(x, y, v, heatmap_size / 2, f, r, s, tx, ty)

    for i in range(total_keypoints):
        # labeled keypoints' v > 0
        if v[i] > 0:
            # ground truth in heatmap is normal distribution shaped
            heatmap[i] = np.exp(-((x_grid - x[i]) ** 2 + (y_grid - y[i]) ** 2) / (2 * sigma ** 2), dtype='float32')
        else:
            heatmap[i] = empty.copy()

    return heatmap


# return ground truth heatmap of a whole training image (fixed-sized square-shaped, can be augmented)
def get_full_image_heatmap(image, keypoint_anns, augment=True):
    # x-y grids
    x_grid = np.repeat(np.array([range(heatmap_size)]), heatmap_size, axis=0)
    y_grid = np.repeat(np.array([range(heatmap_size)]).transpose(), heatmap_size, axis=1)
    empty = np.zeros([heatmap_size, heatmap_size], dtype='float32')

    # heatmap dimension is (number of keypoints)*(heatmap size)*(heatmap size)
    h = image.get('height')
    w = image.get('width')
    heatmap = np.empty((len(keypoint_anns), total_keypoints, heatmap_size, heatmap_size), dtype='float32')

    if augment:
        # random flip, rotation, scaling, translation
        f, r, s, tx, ty = get_augment_parameters(flip, scale, rotate, translate)

    # create individual heatmaps
    for j, keypoint_ann in enumerate(keypoint_anns, 0):
        # keypoints location (x, y) and visibility (v)
        x, y, v = get_coordinates(keypoint_ann, full_image=True)

        # do heatmap augmentation
        if augment:
            x, y, v = augment_heatmap(x, y, v, heatmap_size / 2, f, r, s, tx, ty)

        for i in range(total_keypoints):
            # labeled keypoints' v > 0
            if v[i] > 0:
                # ground truth in heatmap is normal distribution shaped
                heatmap[j][i] = np.exp(-((x_grid - x[i]) ** 2 + (y_grid - y[i]) ** 2) / (2 * sigma ** 2),
                                       dtype='float32')
            else:
                heatmap[j][i] = empty.copy()

    # sum individual heatmaps
    return heatmap.sum(axis=0).clip(0, 1)


# do heatmap augmentation
def augment_heatmap(x, y, v, heatmap_half, f, a, s, tx, ty):
    """把中心點從左上變成中間"""
    x = x - heatmap_half
    y = y - heatmap_half

    # flip
    if f:
        x = -x

        # when flipped, left and right should be swapped
        x = x[left_right_swap]
        y = y[left_right_swap]
        v = v[left_right_swap]

    # rotation
    sin_a = sin(a)
    cos_a = cos(a)
    x, y = tuple(np.dot(np.array([[cos_a, -sin_a], [sin_a, cos_a]]), np.array([x, y])))

    # scaling
    x = x * s
    y = y * s

    # translation
    x = x + tx + heatmap_half
    y = y + ty + heatmap_half

    return x, y, v


"""
Plot something
"""

# plot a heatmap
def plot_heatmap(heatmap, skeleton=None, image_path=None, caption=None, only_skeleton=False):
    # locate the keypoints (the maximum of each channel)
    heatmap_max = np.amax(np.amax(heatmap, axis=1), axis=1)
    index_max = np.array([np.unravel_index(np.argmax(h), h.shape) for h in heatmap])
    x_keypoint = index_max[:, 1]
    y_keypoint = index_max[:, 0]
    keypoint_show = np.arange(total_keypoints)[heatmap_max > heatmap_threshold]

    # option to plot skeleton
    x_skeleton = []
    y_skeleton = []
    skeleton_show = []
    if skeleton is not None:
        skeleton = skeleton[0:17]
        x_skeleton = x_keypoint[skeleton]
        y_skeleton = y_keypoint[skeleton]
        skeleton_show = [i for i in range(len(skeleton)) if (heatmap_max[skeleton[i]] > heatmap_threshold).all()]

        # only plot keypoints and skeleton
        if only_skeleton:
            plt.imshow(np.ones((64, 64, 3), 'float32'))
            [plt.plot(x_skeleton[i], y_skeleton[i], c=skeleton_colors[i], linewidth=5) for i in skeleton_show]
            [plt.plot(x_keypoint[i], y_keypoint[i], 'o', c=keypoint_colors[i], markersize=10, markeredgecolor='k',
                      markeredgewidth=1) for i in keypoint_show]
            plt.title('pose')
            plt.xlabel(caption)
            return

    # get a heatmap in single image with colors
    heatmap_color = np.empty((total_keypoints, heatmap_size, heatmap_size, 3), dtype='float32')
    for i in range(total_keypoints):
        heatmap_color[i] = np.tile(np.array(matplotlib.colors.to_rgb(keypoint_colors_reverse[i])),
                                   (heatmap_size, heatmap_size, 1))
        for j in range(3):
            heatmap_color[i, :, :, j] = heatmap_color[i, :, :, j] * heatmap[i]
    heatmap_color = 1 - np.amax(heatmap_color, axis=0)

    # plot the heatmap in black-white and the optional training image
    if image_path is not None:
        image = io.imread(image_path)
        plt.subplot(1, 2, 1)
        plt.imshow(heatmap_color)
        if skeleton is not None:
            [plt.plot(x_skeleton[i], y_skeleton[i], c=skeleton_colors[i], linewidth=5) for i in skeleton_show]
            [plt.plot(x_keypoint[i], y_keypoint[i], 'o', c=keypoint_colors[i], markersize=10, markeredgecolor='k',
                      markeredgewidth=1) for i in keypoint_show]
        plt.title('stacked heatmaps' + (' and skeleton' if skeleton is not None else ''))
        plt.xlabel(caption)
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title('training image')  
        """im = Image.fromarray((image * 255).astype(np.uint8))
        im.save("yoyoyoitsheatmap.jpeg")"""
    else:        
        plt.imshow(heatmap_color)
        """im = Image.fromarray((heatmap_color * 255).astype(np.uint8))
        im.save("yoyoyoitsheatmap.jpeg")"""
        if skeleton is not None:
            [plt.plot(x_skeleton[i], y_skeleton[i], c=skeleton_colors[i], linewidth=5) for i in skeleton_show]
            [plt.plot(x_keypoint[i], y_keypoint[i], 'o', c=keypoint_colors[i], markersize=10, markeredgecolor='k',
                      markeredgewidth=1) for i in keypoint_show]
        plt.title('stacked heatmaps' + (' and skeleton' if skeleton is not None else ''))
        plt.xlabel(caption)

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