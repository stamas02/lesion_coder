from torchvision import transforms
from PIL import Image
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np


def criterion(x, y):
    return torch.mean(torch.square(torch.subtract(x, y)))

#criterion = nn.L1Loss()

def class_decoder_loss(x, y, label):
    reconstruction_losses = []
    for example_idx, _ in enumerate(label):
        for decoder_idx, _ in enumerate(x):
            if label[example_idx] == decoder_idx:
                reconstruction_losses.append(criterion(x[decoder_idx][example_idx], y[example_idx]))
            else:
                reconstruction_losses.append(criterion(x[decoder_idx][example_idx].detach(), y[example_idx]))
                #econstruction_losses.append(torch.tensor(100).cuda())

    reconstruction_losses = torch.reshape(torch.stack(reconstruction_losses), (len(label), len(x)))
    #reconstruction_losses = torch.t(reconstruction_losses)
    reconstruction_losses = torch.subtract(reconstruction_losses, 0.4)
    reconstruction_losses = torch.mul(reconstruction_losses, 10)
    loss = F.cross_entropy(torch.mul(reconstruction_losses, -1), label)
    predictions = torch.min(reconstruction_losses, 1).indices.detach().cpu().numpy()
    return loss, predictions

def decoder_loss(x, y, label):
    reconstruction_losses = []
    reconstruction_losses_all = []
    for example_idx, _ in enumerate(label):
        for decoder_idx, _ in enumerate(x):
            if label[example_idx] == decoder_idx:
                reconstruction_losses_all.append(criterion(x[decoder_idx][example_idx], y[example_idx]))
                reconstruction_losses.append(criterion(x[decoder_idx][example_idx], y[example_idx]))
            else:
                reconstruction_losses_all.append(criterion(x[decoder_idx][example_idx].detach(), y[example_idx]))

    reconstruction_losses_all = torch.reshape(torch.stack(reconstruction_losses_all), (len(label), len(x)))
    predictions = torch.min(reconstruction_losses_all, 1).indices.detach().cpu().numpy()
    return torch.mean(torch.stack(reconstruction_losses)), predictions




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