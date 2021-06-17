import torch
import random
from hyperparams import beta

from hyperparams import *
from path import *
from utils import *


# a dataset that constructs heatmaps and optional matching caption encodings tensors on the fly
class HeatmapDataset(torch.utils.data.Dataset):
    # a dataset contains keypoints and captions, can add sentence encoding
    def __init__(self, coco_keypoint, coco_caption, single_person=False, text_model=None, full_image=False,
                 for_regression=False):

        # get all containing 'person' image ids
        image_ids = coco_keypoint.getImgIds()

        self.with_vector = (text_model is not None)
        self.for_regression = for_regression
        
        self.full_image = full_image and not for_regression
        self.dataset = []

        for image_id in image_ids:
            """keypoints是一个长度为3k的数组，其中k是category 中 keypoints 的总数量。每一个keypoint 是一个长度为3的数组，
            第一和第二个元素分别是 x和 y坐标值，第三个元素是个标志位 v，v为 0时表示这个关键点没有标注（这种情况下 x = y = v = 0 ），
            v 为 1时表示这个关键点标注了但是不可见（被遮挡了），v 为 2时表示这个关键点标注了同时也可见。
            
            以人類來說category id=1, k為17 依序為0:鼻子,1:左眼,2:右眼...
            以目前看到的例子來說 在human pose的訓練資料裡每張圖片可以有多個annotations，每個annotation會有一個category及其對應的一組keypoints
            然後一張圖片也可能有很多annotations其category都是相同的，好比說一張照片裡有很多人
            也可以說一個category為1的annotation對應到一個人
            """
            keypoint_ann_ids = coco_keypoint.getAnnIds(imgIds=image_id)
            """
            len(keypoint_ann_ids) > 0 表示這張圖片至少有標記了一個東西
            這裡keypoint_ann_ids = coco_keypoint.getAnnIds(imgIds=image_id) 因為是從person_keypoints這個json檔抓進來的 所以應該所有category都是1(human)
            
            """
            if len(keypoint_ann_ids) > 0 and ((single_person and len(keypoint_ann_ids) == 1) or (not single_person)):
                keypoint_anns = coco_keypoint.loadAnns(ids=keypoint_ann_ids)
                caption_ann_ids = coco_caption.getAnnIds(imgIds=image_id)
                caption_anns = coco_caption.loadAnns(ids=caption_ann_ids)
                
                """以整張圖為單位 一筆data代表這張圖裡的所有人。另一個為以個人為單位，一筆data表示一個人"""
                if full_image:
                    data = {'keypoints': [], 'caption': caption_anns.copy(),
                            'image': coco_keypoint.loadImgs(image_id)[0]}
                    """多人的情況下"""
                    for keypoint_ann in keypoint_anns: 
                        """每一個人有17個特徵點，但有些可能沒有標到，這裡是說一個人要標到一定的量才把這個人算進去"""
                        if keypoint_ann.get('num_keypoints') > keypoint_threshold:
                            data['keypoints'].append(keypoint_ann.copy())
                    """這張照片每個人被標的特徵點數量都不達標當作沒人，跳過"""
                    if len(data['keypoints']) == 0:
                        continue

                    # add sentence encoding
                    if text_model is not None:
                        data['vector'] = [get_caption_vector(text_model, caption.get('caption')) for caption in caption_anns]
                    self.dataset.append(data)                    
                else:
                    # each person in the image
                    for keypoint_ann in keypoint_anns:
                        # with enough keypoints
                        if keypoint_ann.get('num_keypoints') > keypoint_threshold:
                            data = {'keypoint': keypoint_ann.copy(), 'caption': caption_anns.copy(),
                                    'image': coco_keypoint.loadImgs(image_id)[0]}

                            # add sentence encoding
                            if text_model is not None:
                                data['vector'] = [get_caption_vector(text_model, caption.get('caption')) for caption in caption_anns]
                            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    # return either individual heatmap of heatmap of a whole image
    def get_heatmap(self, data, augment=True):
        if self.full_image:
            return get_full_image_heatmap(data.get('image'), data.get('keypoints'), augment)
        else:
            return get_single_person_heatmap(data.get('keypoint'), augment)

    def __getitem__(self, index):
        data = self.dataset[index]
        item = dict()

        """if self.for_regression:
            item['coordinates'] = torch.tensor(get_augmented_coordinates(data.get('keypoint')), dtype=torch.float32)
        else:
            # change heatmap range from [0,1] to[-1,1]
            item['heatmap'] = torch.tensor(self.get_heatmap(data) * 2 - 1, dtype=torch.float32)"""
        # change heatmap range from [0,1] to[-1,1]
        item['heatmap'] = torch.tensor(self.get_heatmap(data) * 2 - 1, dtype=torch.float32)

        if self.with_vector:
            # randomly select from all matching captions
            item['vector'] = torch.tensor(random.choice(data.get('vector')), dtype=torch.float32)
            if not self.for_regression:
                """unsqueeze_ 是in place operation, unsqueeze不是"""
                item['vector'].unsqueeze_(-1).unsqueeze_(-1)
        return item

    # get a batch of random caption sentence vectors from the whole dataset
    def get_random_caption_tensor(self, number):
        vector_tensor = torch.empty((number, sentence_vector_size), dtype=torch.float32)

        if self.with_vector:
            for i in range(number):
                # randomly select from all captions
                vector = random.choice(random.choice(self.dataset).get('vector'))
                vector_tensor[i] = torch.tensor(vector, dtype=torch.float32)

        if self.for_regression:
            return vector_tensor
        else:
            return vector_tensor.unsqueeze_(-1).unsqueeze_(-1)

    # get a batch of random heatmaps and captions from the whole dataset
    def get_random_heatmap_with_caption(self, number):
        caption = []
        heatmap = torch.empty((number, total_keypoints, heatmap_size, heatmap_size), dtype=torch.float32)

        for i in range(number):
            # randomly select from all images
            data = random.choice(self.dataset)
            heatmap[i] = torch.tensor(self.get_heatmap(data, augment=False) * 2 - 1, dtype=torch.float32)
            caption.append(random.choice(data.get('caption')).get('caption'))

        return {'heatmap': heatmap, 'caption': caption}

    """完全沒用到R
    # get a batch of random coordinates and captions from the whole dataset
    def get_random_coordinates_with_caption(self, number):
        caption = []
        coordinates = torch.empty((number, total_keypoints * 3), dtype=torch.float32)

        for i in range(number):
            # randomly select from all images
            data = random.choice(self.dataset)
            coordinates[i] = torch.tensor(get_augmented_coordinates(data.get('keypoint')), dtype=torch.float32)
            caption.append(random.choice(data.get('caption')).get('caption'))

        return {'coordinates': coordinates, 'caption': caption}"""

    # get a batch of random interpolated caption sentence vectors from the whole dataset
    def get_interpolated_caption_tensor(self, batch_size):
        vector_tensor = torch.empty((batch_size, sentence_vector_size), dtype=torch.float32)

        if self.with_vector:
            for i in range(batch_size):
                # randomly select 2 captions from all captions
                vector = random.choice(random.choice(self.dataset).get('vector'))
                vector2 = random.choice(random.choice(self.dataset).get('vector'))

                # interpolate caption sentence vectors
                interpolated_vector = beta * vector + (1 - beta) * vector2
                vector_tensor[i] = torch.tensor(interpolated_vector, dtype=torch.float32)
        if self.for_regression:
            return vector_tensor
        else:
            return vector_tensor.unsqueeze_(-1).unsqueeze_(-1)