import os
import numpy as np
from PIL import Image
from skimage import transform, color
import pickle
import torch
from torch.utils.data import Dataset
from segm.model.utils import SoftEncodeAB, CIELAB
import argparse

class ImageNet_dataset(Dataset):
    def __init__(self, dataset_dir, img_size=256, split='train'):
        super(ImageNet_dataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(self.dataset_dir, split)
        self.image_size = img_size
        self.split = split
        self.filenames = self.load_filenames(self.dataset_dir, split)
        self.n_cls = 313

    def load_filenames(self, data_dir, split, filepath='fullfilenames.pickle'):
        if split == 'train':
            split_filepth = os.path.join(data_dir, 'clean_train_filenames.pickle')
        else:
            split_filepth = os.path.join(data_dir, split + '_' + filepath)
        f = open(split_filepth, 'rb')
        filenames = pickle.load(f)
        print('Load from:', split_filepth)
        return filenames

    def resize(self, img, input_size):
        downscale = img.shape[0] > input_size and img.shape[1] > input_size
        res = transform.resize(img, (input_size, input_size), mode='reflect', anti_aliasing=downscale)   # downscale
        if img.dtype == np.uint8:
            res *= 255
        else:
            print('input img.dtype is not np.uint8')
            raise NotImplementedError
        return res.astype(img.dtype)

    def rgb_to_lab(self, img):
        assert img.dtype == np.uint8
        return color.rgb2lab(img).astype(np.float32)

    def numpy_to_torch(self, img):
        tensor = torch.from_numpy(np.moveaxis(img, -1, 0))      # [c, h, w]
        return tensor.type(torch.float32)

    def get_ab(self, key):
        if self.split == 'train':
            img_pth = os.path.join(self.image_dir, key[0], key[1])
        else:
            img_pth = os.path.join(self.image_dir, key)
        img = Image.open(img_pth).convert('RGB')
        img = np.array(img)
        img_resized = self.resize(img, self.image_size)
        ab_resized = self.rgb_to_lab(img_resized)[:, :, 1:]     # np.float32
        img_ab = self.numpy_to_torch(ab_resized)
        return img_ab


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        key = self.filenames[idx]
        img_ab = self.get_ab(key)
        return img_ab, key


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('--dataset_dir', type=str, default='/userhome/SUN_text2img/ImageNet')
    parser.add_argument('--split', type=str, default='val')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset_dir = args.dataset_dir
    split = args.split
    default_cielab = CIELAB()
    encode_ab = SoftEncodeAB(default_cielab)
    if split == 'val':
        val_dataset = ImageNet_dataset(dataset_dir, split='val')
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
        val_dict = {}
        for i, data in enumerate(val_loader):
            print('i', i)
            img_ab, key = data
            img_ab = img_ab.cuda()
            key = key[0]
            q_actual = encode_ab(img_ab)
            np_q_actual = q_actual[0].cpu().detach().numpy()
            val_dict[key] = np_q_actual
        val_fp = open(os.path.join(dataset_dir, 'val_q_actual.pickle'), 'wb')
        pickle.dump(val_dict, val_fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        train_dataset = ImageNet_dataset(dataset_dir, split='train')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        train_dict = {}
        for i, data in enumerate(train_loader):
            print(i)
            img_ab, key = data
            img_ab = img_ab.cuda()
            q_actual = encode_ab(img_ab)
            np_q_actual = q_actual[0].cpu().detach().numpy()
            train_key = key[0][0] + '_' + key[1][0]
            train_dict[train_key] = np_q_actual
        train_fp = open(os.path.join(dataset_dir, 'train_q_actual.pickle'), 'wb')
        pickle.dump(train_dict, train_fp, protocol=pickle.HIGHEST_PROTOCOL)

















