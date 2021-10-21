from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch


class TopTracker():
    def __init__(self, top_k=5, extrema="min"):
        self.items = []
        self.scores = []
        self.k = top_k
        self.extrema = extrema

        self.test = lambda a, b: a < b

    def get_top_items(self):
        if self.extrema == "min":
            sorted_idx = np.argsort(self.scores)
            return [self.items[i] for i in sorted_idx]
        elif self.extrema == "max":
            sorted_idx = np.argsort(self.scores)[::-1]
            return [self.items[i] for i in sorted_idx]

    def add(self, score, item):
        if len(self.items) < self.k:
            self.items.append(item)
            self.scores.append(score)
        else:
            if self.extrema == "min":
                max_score_ind = np.argmax(self.scores)
                if self.scores[max_score_ind] > score:
                    self.scores[max_score_ind] = score
                    self.items[max_score_ind] = item
            elif self.extrema == "max":
                min_score_ind = np.argmin(self.scores)
                if self.scores[min_score_ind] > score:
                    self.scores[min_score_ind] = score
                    self.items[min_score_ind] = item


def get_train_transform(input_size):
    """

    :return:
    """

    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([transforms.RandomAffine(180,
                                                         scale=(0.8, 1.2),
                                                         shear=10,
                                                         resample=Image.NEAREST),
                                 transforms.RandomAffine(180,
                                                         scale=(0.8, 1.2),
                                                         shear=10,
                                                         resample=Image.BICUBIC),
                                 transforms.RandomAffine(180,
                                                         scale=(0.8, 1.2),
                                                         shear=10,
                                                         resample=Image.BILINEAR)]),
        transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def get_test_transform(input_size):
    """

    :return:
    """

    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def get_denormalize_transform():
    m = 0.5
    s = 0.5
    return transforms.Compose([
        transforms.Normalize(mean=[-m / s, -m / s, -m / s], std=[1 / s, 1 / s, 1 / s])]
    )



def slit_data(df, test_split, val_split, seed=7):
    indices = np.array(range(df.shape[0]))
    np.random.seed(seed)
    np.random.shuffle(indices)
    split_point_1 = int(indices.shape[0] * test_split)
    split_point_2 = int(indices.shape[0] * (val_split + test_split))
    test_indices = indices[0:split_point_1]
    val_indices = indices[split_point_1:split_point_2]
    train_indices = indices[split_point_2::]
    train_df = df.take(train_indices)
    test_df = df.take(test_indices)
    val_df = df.take(val_indices)
    return train_df, test_df, val_df
