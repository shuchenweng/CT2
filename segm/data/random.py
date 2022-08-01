import os
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from skimage.io import imread
# from skimage import transform, color
from skimage import color

from segm.data import utils
import pickle
import torch
from segm.config import dataset_dir
# import cv2
import collections
import json


class RandomDataset(Dataset):
    def __init__(
        self,
        image_size=256,
        crop_size=256,
        split="val",
        normalization="vit",
        dataset_dir='',
        add_mask=False,
        patch_size=16,
        change_mask=False,
        multi_scaled=False,
        mask_num=4,
    ):
        super().__init__()
        self.dataset_dir = dataset_di
        self.crop_size = crop_size
        self.image_size = image_size
        self.split = split
        self.add_mask = add_mask
        self.patch_size = patch_size
        self.change_mask = change_mask
        self.multi_scaled = multi_scaled
        self.mask_num = mask_num
        assert self.crop_size % self.patch_size == 0

        self.filenames = self.load_filenames(self.dataset_dir, split)
        self.n_cls = 313
        if self.add_mask:
            assert os.path.exists(os.path.join(self.dataset_dir, 'mask_prior.pickle'))
            fp = open(os.path.join(self.dataset_dir, 'mask_prior.pickle'), 'rb')
            L_dict = pickle.load(fp)

            self.mask_L = np.zeros((mask_num, 313)).astype(np.bool)     # [4, 313]
            for key in range(101):
                for ii in range(mask_num):
                    start_key = ii * (100//mask_num)      # 0
                    end_key = (ii+1)* (100//mask_num)     # 25
                    if start_key <= key < end_key:
                        self.mask_L[ii, :] += L_dict[key].astype(np.bool)
                        break

            self.mask_L = self.mask_L.astype(np.float32)
            del L_dict

    @property
    def unwrapped(self):
        return self

    def load_filenames(self, data_dir, split, filepath='fullfilenames.pickle'):
        filenames = os.listdir(data_dir)
        return filenames

    def rgb_to_lab(self, img):
        assert img.dtype == np.uint8
        return color.rgb2lab(img).astype(np.float32)

    def numpy_to_torch(self, img):
        tensor = torch.from_numpy(np.moveaxis(img, -1, 0))      # [c, h, w]
        return tensor.type(torch.float32)

    def get_img(self, key):
        img_pth = os.path.join(self.dataset_dir, key)
        img = Image.open(img_pth).convert("RGB")
        w, h = img.size
        if w != 256 or h != 256:
            mini_size = min(w, h)
            img_transform = transforms.Compose([
                transforms.CenterCrop(mini_size),
                transforms.Resize(self.crop_size),
            ])
            img = img_transform(img)

        img_resized = np.array(img)
        l_resized = self.rgb_to_lab(img_resized)[:, :, :1]
        ab_resized = self.rgb_to_lab(img_resized)[:, :, 1:]     # np.float32

        mask = torch.ones(1)
        if self.add_mask:
            original_l = l_resized[:, :, 0]
            l = original_l.reshape((self.crop_size * self.crop_size))
            mask_p_c = np.zeros((self.crop_size**2, self.n_cls), dtype=np.float32)

            for l_range in range(self.mask_num):
                start_l1, end_l1 = l_range * (100//self.mask_num), (l_range + 1) * (100 // self.mask_num)
                if end_l1 == 100:
                    index_l1 = np.where((l >= start_l1) & (l <= end_l1))[0]
                else:
                    index_l1 = np.where((l >= start_l1) & (l < end_l1))[0]
                mask_p_c[index_l1, :] = self.mask_L[l_range, :]

            mask = torch.from_numpy(mask_p_c)
        img_l = self.numpy_to_torch(l_resized)
        img_ab = self.numpy_to_torch(ab_resized)

        return img_l, img_ab, mask

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        key = self.filenames[idx]
        img_l, img_ab, mask = self.get_img(key)
        return img_l, img_ab, key, mask

