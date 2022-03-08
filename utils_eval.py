import numpy as np
import torch
from tqdm import tqdm


def pose_distance(pose_1, pose_2):
    return torch.sum(torch.sqrt(torch.sum((pose_1 - pose_2) ** 2, axis=1)))

    #return torch.sum(torch.sqrt(torch.sum((pose_1-pose_2)**2, dim=-1)), dim=-1).mean()

# find the nearest neighbor distance of a heatmap in a list of heatmaps
def nearest_neighbor_distance(pose, pose_list):
    distance = pose_distance(pose, pose_list[0])
    # find nearest neighbor
    for another_pose in pose_list[1:]:
        new_distance = pose_distance(pose, another_pose)
        if new_distance < distance:
            distance = new_distance
    return distance

# calculate the one-nearest-neighbor accuracy
def one_nearest_neighbor(real_pose_list, fake_pose_list):
    # number of correct classifications
    count = 0
    for i in tqdm(range(len(real_pose_list)), desc='  - (Dataset)   ', leave=True):
        # a heatmap from the first list
        x = nearest_neighbor_distance(real_pose_list[i], real_pose_list[0:i] + real_pose_list[i + 1:])
        y = nearest_neighbor_distance(real_pose_list[i], fake_pose_list)
        if x < y:
            count = count + 1

        x = nearest_neighbor_distance(fake_pose_list[i], fake_pose_list[0:i] + fake_pose_list[i + 1:])
        y = nearest_neighbor_distance(fake_pose_list[i], real_pose_list)        
        if x < y:
            count = count + 1
    # accuracy
    return count / len(real_pose_list) / 2


if __name__ == "__main__":
    a = torch.tensor([[[2,2,2,2],[3,3,3,3]],[[4,4,4,4],[5,5,5,5]]], dtype=torch.float32)
    b = torch.tensor([[[1,1,1,1],[1,1,1,1]],[[1,1,1,1],[1,1,1,1]]], dtype=torch.float32)
    print(a.size())
    print((a-b)**2)
    print(torch.sum((a-b)**2, dim=-1))
    print(torch.sqrt(torch.sum((a-b)**2, dim=-1)))
    print(torch.sqrt(torch.sum((a-b)**2, dim=-1)).mean(-1))
