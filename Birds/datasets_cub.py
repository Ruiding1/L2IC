import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import random
import numpy as np
from config import cfg
import os
import os.path
import pandas as pd
import torch
import cv2
import mat4py
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import yaml


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            ##  self._dict = yaml.load(self._yaml)
            self._dict = yaml.load(self._yaml, Loader=yaml.FullLoader)
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


class CubDataset(Dataset):
    def __init__(
            self,
            config,
            data_split=0,
            use_flip=True
    ):
        super().__init__()
        self.data_split = data_split
        self.use_flip = use_flip

        self.transform = transforms.Compose([
            transforms.Resize((int(config.IM_SIZE), int(config.IM_SIZE))),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.transform_seg = transforms.Compose([
            transforms.Resize((int(config.IM_SIZE), int(config.IM_SIZE))),
            transforms.ToTensor()])

        self.ROOT_DIR = config.ROOT_DIR
        self.IM_SIZE = config.IM_SIZE

        self.bbox_meta, self.file_meta = self.collect_meta()

    def __len__(self):
        return len(self.file_meta)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: sample # {}'.format(index))
            item = self.load_item(0)

        return item

    def collect_meta(self):
        """ Returns a dictionary with image filename as
        'key' and its bounding box coordinates as 'value' """

        data_dir = self.ROOT_DIR

        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)

        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)

        filenames = df_filenames[1].tolist()
        # print('Total filenames: ', len(filenames), filenames[0])
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)

        splits = np.loadtxt(os.path.join(data_dir, 'train_val_test_split.txt'), int)

        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox

        filenames = [fname[:-4] for fname in filenames]

        if self.data_split == 0:
            filenames = np.array(filenames)
            filenames = filenames[splits[:, 1] == 0]
            filename_bbox_ = {fname: filename_bbox[fname] for fname in filenames}
        elif self.data_split == 2:
            filenames = np.array(filenames)
            filenames = filenames[splits[:, 1] == 2]
            filename_bbox_ = {fname: filename_bbox[fname] for fname in filenames}
        elif self.data_split == -1:
            filenames = filenames.copy()
            filename_bbox_ = filename_bbox

        print('Filtered filenames: ', len(filenames))
        return filename_bbox_, filenames

    def load_item(self, index):
        key = self.file_meta[index]
        bbox = self.bbox_meta[key]

        data_dir = self.ROOT_DIR

        img_path = '%s/images/%s.jpg' % (data_dir, key)
        img = self.load_imgs(img_path, bbox)

        seg_path = '%s/segmentations/%s.png' % (data_dir, key)
        seg = self.load_segs(seg_path, bbox)

        if self.use_flip and np.random.uniform() > 0.5:
            img = torch.flip(img, dims=[-1])
            seg = torch.flip(seg, dims=[-1])
        return img, seg, index

    def load_imgs(self, img_path, bbox):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)

        cimg = img.crop([x1, y1, x2, y2])
        return self.transform(cimg)

    def load_segs(self, seg_path, bbox):
        img = Image.open(seg_path)#.convert('1')
        width, height = img.size

        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)

        cimg = img.crop([x1, y1, x2, y2])
        cimg = self.transform_seg(cimg)
        if cimg.size(0) != 1:
            cimg = cimg[:1]
        cimg = (cimg > .5).float()
        return cimg

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=False
            )

            for item in sample_loader:
                yield item

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        fimg = deepcopy(img)
        fimg_arr = np.array(fimg)
        fimg = Image.fromarray(fimg_arr)
        cimg = img.crop([x1, y1, x2, y2])

    if transform is not None:
        cimg = transform(cimg)

    retf = []
    retc = []
    re_cimg = transforms.Resize(imsize[1])(cimg)
    retc.append(normalize(re_cimg))

    transform_aug = transforms.Compose([transforms.RandomResizedCrop(128, scale=(0.2, 1.0)),
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

    my_crop_width = 126
    re_fimg = transforms.Resize(int(my_crop_width * 76 / 64))(fimg)
    re_width, re_height = re_fimg.size

    # random cropping
    x_crop_range = re_width - my_crop_width
    y_crop_range = re_height - my_crop_width

    crop_start_x = np.random.randint(x_crop_range)
    crop_start_y = np.random.randint(y_crop_range)

    crop_re_fimg = re_fimg.crop([crop_start_x, crop_start_y, crop_start_x + my_crop_width, crop_start_y + my_crop_width])
    warped_x1 = bbox[0] * re_width / width
    warped_y1 = bbox[1] * re_height / height
    warped_x2 = warped_x1 + (bbox[2] * re_width / width)
    warped_y2 = warped_y1 + (bbox[3] * re_height / height)

    warped_x1 = min(max(0, warped_x1 - crop_start_x), my_crop_width)
    warped_y1 = min(max(0, warped_y1 - crop_start_y), my_crop_width)
    warped_x2 = max(min(my_crop_width, warped_x2 - crop_start_x), 0)
    warped_y2 = max(min(my_crop_width, warped_y2 - crop_start_y), 0)

    # random flipping
    random_flag = np.random.randint(2)
    if (random_flag == 0):
        crop_re_fimg = crop_re_fimg.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_x1 = my_crop_width - warped_x2
        flipped_x2 = my_crop_width - warped_x1
        warped_x1 = flipped_x1
        warped_x2 = flipped_x2

    retf.append(normalize(crop_re_fimg))

    warped_bbox = []
    warped_bbox.append(warped_y1)
    warped_bbox.append(warped_x1)
    warped_bbox.append(warped_y2)
    warped_bbox.append(warped_x2)

    return retf[0], retc[0], warped_bbox


class Dataset_FineGAN(data.Dataset):
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
        self.bbox = self.load_bbox()
        self.filenames = self.load_filenames(data_dir)

        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    # only used in background stage

    def load_bbox(self):
        # Returns a dictionary with image filename as 'key' and its bounding box coordinates as 'value'
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        # print('Total filenames: ', len(filenames), filenames[0])
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        return filename_bbox

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        filenames = [fname[:-4] for fname in filenames]
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        data_dir = self.data_dir
        img_name = '%s/images/%s.jpg' % (data_dir, key)

        fimgs, cimgs, warped_bbox = get_imgs(img_name, self.imsize,
                                             bbox, self.transform, normalize=self.norm)

        # Randomly generating child code during training
        rand_class = random.sample(range(cfg.FINE_GRAINED_CATEGORIES), 1)
        c_code = torch.zeros([cfg.FINE_GRAINED_CATEGORIES, ])
        c_code[rand_class] = 1
        return fimgs, cimgs, c_code, key, warped_bbox

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        data_dir = self.data_dir
        c_code = self.c_code[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        _, imgs, _ = get_imgs(img_name, self.imsize,
                              bbox, self.transform, normalize=self.norm)

        return imgs, c_code, key

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)


def get_dataloader(bs=None):
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))

    image_transform = transforms.Compose([transforms.Resize(int(imsize * 76 / 64)),
                                          transforms.RandomCrop(imsize),
                                          transforms.RandomHorizontalFlip()])

    dataset = Dataset_FineGAN(cfg.DATA_DIR, base_size=cfg.TREE.BASE_SIZE, transform=image_transform)

    if bs == None:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=True, shuffle=True)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=True, shuffle=True)

    return dataset, dataloader


class Dataset(data.Dataset):
    def __init__(self, data_dir, mode='train'):

        self.transform = transforms.Compose([transforms.Resize(128),
                                             transforms.RandomCrop(128),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ])

        self.transform_aug = transforms.Compose([transforms.RandomResizedCrop(128, scale=(0.2, 1.0)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                                 transforms.RandomGrayscale(0.2),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ])
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        self.data_dir = data_dir
        self.img_paths, self.labels = self.load_filenames(data_dir, mode)

        if mode == 'train':
            self.iterator = self.prepare_training_pairs
        else:
            self.iterator = self.prepare_test_pairs

    def load_filenames(self, data_dir, mode):
        if mode == 'train':
            with open(os.path.join(data_dir, 'trainset.txt'), 'r') as f:
                data = f.readlines()
        else:
            with open(os.path.join(data_dir, 'testset.txt'), 'r') as f:
                data = f.readlines()
        img_paths = [os.path.join(data_dir, 'images', _.split()[0]) for _ in data]
        labels = [int(_.split()[-1]) for _ in data]
        return img_paths, labels

    def prepare_training_pairs(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img_ = self.transform(img)
        img_aug_ = self.transform_aug(img)
        return img_, img_aug_

    def prepare_test_pairs(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = transforms.Resize(128)(img)
        img = transforms.CenterCrop(128)(img)
        img = self.norm(img)
        return img, self.labels[index]

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.img_paths)


class Dataset_10class(data.Dataset):
    def __init__(self, data_dir, mode='train'):

        self.transform = transforms.Compose([transforms.Resize(128),
                                             transforms.RandomCrop(128),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ])

        self.transform_aug = transforms.Compose([transforms.RandomResizedCrop(128, scale=(0.2, 1.0)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                                 transforms.RandomGrayscale(0.2),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ])
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        self.data_dir = data_dir
        self.img_paths, self.labels = self.load_filenames(data_dir, mode)

        if mode == 'train':
            self.iterator = self.prepare_training_pairs
        else:
            self.iterator = self.prepare_test_pairs

    def load_filenames(self, data_dir, mode):
        if mode == 'train':
            with open(os.path.join(data_dir, 'trainset_10class.txt'), 'r') as f:
                data = f.readlines()
        else:
            with open(os.path.join(data_dir, 'testset_10class.txt'), 'r') as f:
                data = f.readlines()
        img_paths = [os.path.join(data_dir, 'images', _.split()[0]) for _ in data]
        labels = [int(_.split()[-1]) for _ in data]
        return img_paths, labels

    def prepare_training_pairs(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img_ = self.transform(img)
        img_aug_ = self.transform_aug(img)
        return img_, img_aug_

    def prepare_test_pairs(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = transforms.Resize(128)(img)
        img = transforms.CenterCrop(128)(img)
        img = self.norm(img)
        return img, self.labels[index], self.img_paths[index]

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.img_paths)



