import glob
import os.path as osp
import random
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torch.utils.data as data


class ImageTransform:
    """
    image pre-processing: resize image, normalization RGB value
    version : train, validation
    * train_version : image data augmentation

    Attributes
    ----------
    resize: int
    mean : (R, G, B)
    std : (R, G, B)
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        """
        :param img: image data
        :param phase: 'train' or 'val' -> specify dataset mode
        :return: self.data_transform
        """
        return self.data_transform[phase](img)


def create_data_path(phase="train"):
    """
    make data path list
    :param
    phase: 'train' or 'val'
    :return:
    path_list: list
    """
    root_path = './data/'
    target_path = osp.join(root_path + phase + '/**/*.jpg')

    path_list = []
    # glob -> load file path of sub directory
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


class ProductDataset(data.Dataset):
    """
    ant, bee image Dataset class. Dataset class 상속

    Attributes
    ----------
    file_list : list
        -> file path list
    transform : object
        -> data pre-processing instance
    phase : 'train' or 'val'

    """

    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        """return length of images"""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        get the Tensor and label of pre-processed image
        :param idx: index of data
        :return:
        """

        # load image
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        # pre-processing image data
        img_transformed = self.transform(img, self.phase)  # torch.Size([3, 224, 224])

        # get the image label from file path(name)
        if "ants" in img_path:
            label = 0
        elif "bees" in img_path:
            label = 1

        return img_transformed, label
