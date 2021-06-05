import torch
import random
from math import beta

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
        if for_regression:
            full_image = False
        self.full_image = full_image
        self.dataset = []

        for image_id in image_ids:
            keypoint_ids = coco_keypoint.getAnnIds(imgIds=image_id)
            if len(keypoint_ids) > 0 and ((single_person and len(keypoint_ids) == 1) or (not single_person)):
                caption_ids = coco_caption.getAnnIds(imgIds=image_id)
                captions = coco_caption.loadAnns(ids=caption_ids)
                keypoints = coco_keypoint.loadAnns(ids=keypoint_ids)

                if full_image:
                    data = {'keypoints': [], 'caption': captions.copy(),
                            'image': coco_keypoint.loadImgs(image_id)[0]}
                    for keypoint in keypoints:
                        if keypoint.get('num_keypoints') > keypoint_threshold:
                            data['keypoints'].append(keypoint.copy())
                    if len(data['keypoints']) == 0:
                        continue

                    # add sentence encoding
                    if text_model is not None:
                        data['vector'] = [get_caption_vector(text_model, caption.get('caption')) for caption in
                                          captions]
                    self.dataset.append(data)
                else:
                    # each person in the image
                    for keypoint in keypoints:
                        # with enough keypoints
                        if keypoint.get('num_keypoints') > keypoint_threshold:
                            data = {'keypoint': keypoint.copy(), 'caption': captions.copy(),
                                    'image': coco_keypoint.loadImgs(image_id)[0]}

                            # add sentence encoding
                            if text_model is not None:
                                data['vector'] = [get_caption_vector(text_model, caption.get('caption')) for caption in
                                                  captions]
                            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    # return either individual heatmap of heatmap of a whole image
    def get_heatmap(self, data, augment=True):
        if self.full_image:
            return get_full_image_heatmap(data.get('image'), data.get('keypoints'), augment)
        else:
            return get_heatmap(data.get('keypoint'), augment)

    def __getitem__(self, index):
        data = self.dataset[index]
        item = dict()

        if self.for_regression:
            item['coordinates'] = torch.tensor(get_augmented_coordinates(data.get('keypoint')), dtype=torch.float32)
        else:
            # change heatmap range from [0,1] to[-1,1]
            item['heatmap'] = torch.tensor(self.get_heatmap(data) * 2 - 1, dtype=torch.float32)

        if self.with_vector:
            # randomly select from all matching captions
            item['vector'] = torch.tensor(random.choice(data.get('vector')), dtype=torch.float32)
            if not self.for_regression:
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

    # get a batch of random coordinates and captions from the whole dataset
    def get_random_coordinates_with_caption(self, number):
        caption = []
        coordinates = torch.empty((number, total_keypoints * 3), dtype=torch.float32)

        for i in range(number):
            # randomly select from all images
            data = random.choice(self.dataset)
            coordinates[i] = torch.tensor(get_augmented_coordinates(data.get('keypoint')), dtype=torch.float32)
            caption.append(random.choice(data.get('caption')).get('caption'))

        return {'coordinates': coordinates, 'caption': caption}

    # get a batch of random interpolated caption sentence vectors from the whole dataset
    def get_interpolated_caption_tensor(self, number):
        vector_tensor = torch.empty((number, sentence_vector_size), dtype=torch.float32)

        if self.with_vector:
            for i in range(number):
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


