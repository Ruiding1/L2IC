import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import random
import numpy as np
from config import cfg
import torch.utils.data as data
import os
import os.path
import pandas as pd
import torch
from copy import deepcopy

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imgs(img_path, imsize, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    cimg = img
    if transform is not None:
        cimg = transform(cimg)

    retf = []
    retc = []
    # re_cimg = transforms.Resize(imsize[1])(cimg)
    re_cimg = transforms.Resize([64,64])(cimg)
    retc.append(normalize(re_cimg))

    transform_aug = transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                        transforms.RandomGrayscale(0.2),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
    retf.append(transform_aug(re_cimg))

    # We use full image to get background patches

    # We resize the full image to be 126 X 126 (instead of 128 X 128)  for the full coverage of the input (full) image by
    # the receptive fields of the final convolution layer of background discriminator

    # my_crop_width = 128
    # re_fimg = transforms.Resize(int(my_crop_width * 76 / 64))(fimg)

    return retf[0], retc[0]


def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in (fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


class Dataset(data.Dataset):
    def __init__(self, data_dir, base_size=64, transform=None):

        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir

        self.filenames = make_dataset(data_dir)
        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_bbox(self):
        # Returns a dictionary with image filename as 'key' and its bounding box coordinates as 'value'

        data_dir = self.data_dir
        files_path = os.path.join(data_dir, 'Annotation')
        for target in os.listdir(files_path):
            d = os.path.join(files_path, target)
            for target in os.listdir(d):
                d_bbox = os.path.join(d, target)
                files = open(d_bbox)
                d_type = os.path.join(d_bbox.split('/', -1)[-2], d_bbox.split('/', -1)[-1])
                files_bbox = files.read()
                files_xmin = files_bbox.split('<xmin>')[1]
                files_xmin = files_xmin.split('</xmin>')[0]
                files_xmin = int(files_xmin)

                files_ymin = files_bbox.split('<ymin>')[1]
                files_ymin = files_ymin.split('</ymin>')[0]
                files_ymin = int(files_ymin)

                files_xmax = files_bbox.split('<xmax>')[1]
                files_xmax = files_xmax.split('</xmax>')[0]
                files_xmax = int(files_xmax)

                files_ymax = files_bbox.split('<ymax>')[1]
                files_ymax = files_ymax.split('</ymax>')[0]
                files_ymax = int(files_ymax)

                bbox = [files_xmin, files_ymin, files_xmax, files_ymax]
                files_all = {d_type: bbox}

                self.bbox_files.update(files_all)

        return self.bbox_files


    def prepair_training_pairs(self, index):

        key_files = self.filenames[index].split("/", -1)[-1].split('.')[0]
        key = os.path.join(self.filenames[index].split("/", -1)[-2], key_files)
        img_name = self.filenames[index]
        fimgs, cimgs = get_imgs(img_name, self.imsize, self.transform, normalize=self.norm)

        rand_class = random.sample(range(cfg.FINE_GRAINED_CATEGORIES), 1)
        c_code = torch.zeros([cfg.FINE_GRAINED_CATEGORIES, ])
        c_code[rand_class] = 1

        return fimgs, cimgs, c_code, key

    def prepair_test_pairs(self, index):
        key = self.filenames[index]

        data_dir = self.data_dir
        c_code = self.c_code[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        _, imgs, _ = get_imgs(img_name, self.imsize, self.transform, normalize=self.norm)

        return imgs, c_code, key

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)


def get_dataloader(bs=None):

    image_transform = transforms.Compose([transforms.Resize(64),
                                          transforms.RandomHorizontalFlip()])

    dataset = Dataset(cfg.DATA_DIR, base_size=cfg.TREE.BASE_SIZE, transform=image_transform)

    if bs == None:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=True, shuffle=True)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=True, shuffle=True)

    return dataset, dataloader










