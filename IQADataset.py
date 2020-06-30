import torch
import os
from torch.utils.data import Dataset
from scipy.signal import convolve2d
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as transforms
import cv2 as cv
import torchvision
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt

def gray_loader(path):
    return Image.open(path).convert('L')


def image_preprocess(patch, P=4, Q=4):
    w, h = patch[0].shape

    kernel = np.ones((P, Q)) / (P * Q)
    patch_low = convolve2d(patch[0], kernel, boundary='symm', mode='same')
    patch_low = torch.from_numpy(patch_low)
    patch_low = patch_low.unsqueeze(0).unsqueeze(0)
    patch_low = F.interpolate(patch_low, size=(w//4, h//4))

    patch_low = F.interpolate(patch_low, size=(w, h))
    patch_low = patch_low.squeeze(0)

    patch = patch - patch_low.float()

    return patch

def CropPatches(image, patch_size=112, stride=80):
    w, h = image.size

    patches = ()
    for i in range(0, h-stride, stride):
        for j in range(0, w-stride, stride):
            patch = to_tensor(image.crop((j, i, j+patch_size, i+patch_size)))
            patch = image_preprocess(patch)
            patches = patches + (patch,)

    return patches


def errormap_process(patch, patch_ref):
    p = 0.2
    errormap = np.abs(patch_ref.numpy(), patch.numpy())
    errormap = np.power(errormap, p)
    errormap = torch.from_numpy(errormap)
    errormap = errormap.unsqueeze(0)
    errormap = F.interpolate(errormap, size=(28, 28))
    errormap = errormap.squeeze(0)
    return errormap

def CropPatches_errormap(image, ref, patch_size=112, stride=80):
    w, h = image.size
    patches_errormap = ()
    for i in range(0, h-stride, stride):
        for j in range(0, w-stride, stride):
            patch = to_tensor(image.crop((j, i, j+patch_size, i+patch_size)))
            patch_ref = to_tensor(ref.crop((j, i, j+patch_size, i+patch_size)))
            patch = errormap_process(patch, patch_ref)
            patches_errormap = patches_errormap + (patch,)

    return patches_errormap

class IQADataset(Dataset):
    def __init__(self, dataset, config, index, status):

        im_dir = config[dataset]['im_dir']
        ref_dir = config[dataset]['ref_dir']

        self.gray_loader = gray_loader
        self.patch_size = config['patch_size']
        self.stride = config['stride']

        test_ratio = config['test_ratio']
        train_ratio = config['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1 - test_ratio) * len(index)):]
        train_index, val_index, test_index = [], [], []

        ref_ids = []
        for line0 in open("./data/live/ref_ids.txt", "r"):
            line0 = float(line0[:-1])
            ref_ids.append(line0)
        ref_ids = np.array(ref_ids)

        for i in range(len(ref_ids)):
            if (ref_ids[i] in trainindex):
                train_index.append(i)
            elif (ref_ids[i] in testindex):
                test_index.append(i)

        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(testindex)


        self.mos = []
        for line5 in open("./data/live/mos.txt", "r"):
            line5 = float(line5.strip())
            self.mos.append(line5)
        self.mos = np.array(self.mos)

        im_names = []
        ref_names = []
        for line1 in open("./data/live/im_names.txt", "r"):
            line1 = line1.strip()
            im_names.append(line1)
        im_names = np.array(im_names)

        for line2 in open("./data/live/refnames.txt", "r"):
            line2 = line2.strip()
            ref_names.append(line2)
        ref_names = np.array(ref_names)

        self.patches = ()
        self.patches_errormap = ()
        self.label = []

        self.im_names = [im_names[i] for i in self.index]
        self.ref_names = [ref_names[i] for i in self.index]
        self.mos = [self.mos[i] for i in self.index]

        for idx in range(len(self.index)):
            # print("{} {} {}".format(self.im_names[idx], self.ref_names[idx], self.mos[idx]))
            im = self.gray_loader(os.path.join(im_dir, self.im_names[idx]))
            ref = self.gray_loader(os.path.join(ref_dir, self.ref_names[idx]))

            patches = CropPatches(im, self.patch_size, self.stride)
            patches_errormap = CropPatches_errormap(im, ref, self.patch_size, self.stride)

            if status == 'train':
                self.patches = self.patches + patches
                self.patches_errormap = self.patches_errormap + patches_errormap
                for i in range(len(patches)):
                    self.label.append(self.mos[idx])
            elif status == 'test':
                self.patches = self.patches + (torch.stack(patches), )
                self.patches_errormap = self.patches_errormap + patches_errormap
                self.label.append(self.mos[idx])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return (self.patches[idx], self.patches_errormap[idx]), (torch.Tensor([self.label[idx]]))




