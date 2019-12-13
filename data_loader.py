import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import cv2

class ImageDataset(Dataset):

    def __init__(self, root, split='train', width=320, height=240, transform=None):
        self.root = root
        self.images_dir = os.path.join(self.root, split)  # train or val
        self.images = []  # image list
        self.targets = []  # label list
        self.width = width
        self.height = height
        self.transform = transform

        for label in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, label)
            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(int(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        image = image.resize((self.width, self.height))
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'target': self.targets[index]}
        return sample


class VOC(Dataset):
    def __init__(self, li, transform=None, size=(224, 224)):
        self.transform = transform
        self.size = size
        self.img = []
        self.lab = []
        for i in li:
            self.img.append(i[0])
            self.lab.append(int(i[1]))

    def __getitem__(self, index):
        img_path = '../../dataset/VOCdevkit/VOC2012/JPEGImages/' + self.img[index] + '.jpg'
        #        print (img_path)
        image = cv2.imread(img_path)
        image = cv2.resize(image, self.size)
        image = self.transform(image)
        label = torch.LongTensor([self.lab[index]])
        sample = {'image': image, 'target': label}
        return sample
        #return image, label

    def __len__(self):
        return len(self.img)