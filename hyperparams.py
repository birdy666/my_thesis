
# ground truth size in heatmap
sigma = 2
# size of heatmap input to network
heatmap_size = int(64)
# heatmap augmentation parameters
flip = 0.5
rotate = 10
scale = 1
translate = 0
""""keypoints": ["nose",
                "left_eye","right_eye","left_ear","right_ear",
                "left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle"]"""
left_right_swap = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
# size of text encoding
sentence_vector_size = 300
# weight of text encoding
encoding_weight = 30
# size of compressed text encoding
compress_size = 128
# text encoding interpolation
beta = 0.5
# numbers of channels of the convolutions
convolution_channel_g = [256, 128, 64, 32]
convolution_channel_d = [32, 64, 128, 256]
# hidden layers
hidden = [250, 200]
# coordinate scaling
factor = 20
# visible threshold
v_threshold = 0.8

noise_size = 128

# have more than this number of keypoints to be included
keypoint_threshold = 7

# to decide whether a keypoint is in the heatmap
heatmap_threshold = 0.2


total_keypoints = 17
keypoint_colors = ['#057020', '#11bb3b', '#12ca3e', '#11bb3b', '#12ca3e', '#1058d1', '#2e73e5', '#cabe12', '#eae053',
                   '#cabe12', '#eae053', '#1058d1', '#2e73e5', '#9dc15c', '#b1cd7e', '#9dc15c', '#b1cd7e']
keypoint_colors_reverse = [hex(0xffffff - int(c.replace('#', '0x'), 16)).replace('0x', '#') for c in keypoint_colors]
skeleton_colors = ['#b0070a', '#b0070a', '#f40b0f', '#f40b0f', '#ec7f18', '#ad590b', '#ef9643', '#ec7f18', '#952fe9',
                   '#b467f4', '#952fe9', '#b467f4', '#ee6da5', '#ee6da5', '#ee6da5', '#c8286e', '#e47ca9', '#c8286e',
                   '#e47ca9']


