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
from scipy import io
import yaml


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

class FlowersDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, sets='train', transform=transforms.ToTensor()):
        super(FlowersDataset, self).__init__()
        self.files = io.loadmat(os.path.join(dataPath, "setid.mat"))
        if sets == 'train':
            self.files = self.files.get('tstid')[0]
        elif sets == 'val':
            self.files = self.files.get('valid')[0]
        else:
            self.files = self.files.get('trnid')[0]
        self.transform = transform
        self.datapath = dataPath
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        imgname = "image_%05d.jpg" % self.files[idx]
        segname = "segmim_%05d.jpg" % self.files[idx]
        img = self.transform(Image.open(os.path.join(self.datapath, "jpg", imgname)))
        seg = np.array(Image.open(os.path.join(self.datapath, "segmim", segname)))
        seg = 1 - ((seg[:,:,0:1] == 0) + (seg[:,:,1:2] == 0) + (seg[:,:,2:3] == 254))
        seg = (seg * 255).astype('uint8').repeat(3,axis=2)
        seg = self.transform(Image.fromarray(seg))[:1]
        return img * 2 - 1, seg


class ClevrDataset(Dataset):
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
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_seg = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()])

        self.ROOT_DIR = config.ROOT_DIR
        self.IM_SIZE = 128
        self.N_OBJ_MAX = 6
        self.N_OBJ_MIN = 3
        self.N = 8000

        self.file_meta = self.collect_meta()

    def __len__(self):
        return len(self.file_meta) if (self.data_split == -1) else 1000

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: sample # {}'.format(index))
            item = self.load_item(0)
        return item

    def collect_meta(self):
        filenames = []
        meta_dir = self.ROOT_DIR

        for i in range(self.N):
            meta_path = '{}/meta/{}.npz'.format(meta_dir, i)
            meta = np.load(meta_path, allow_pickle=True)

            cur_n_obj = meta['visibility'].sum()
            # filter the metas using number of visible objects
            if cur_n_obj <= self.N_OBJ_MAX and \
                    cur_n_obj >= self.N_OBJ_MIN:
                filenames.append(i)

        return filenames

    def load_item(self, index):
        key = self.file_meta[index]

        data_dir = self.ROOT_DIR

        meta_path = '{}/meta/{}.npz'.format(data_dir, key)
        img = self.load_imgs(meta_path)
        seg = self.load_segs(meta_path)

        if self.use_flip and np.random.uniform() > 0.5:
            img = torch.flip(img, dims=[-1])
            seg = torch.flip(seg, dims=[-1])
        return img, seg, index

    def load_imgs(self, meta_path):
        img = np.load(meta_path)['image'][0]
        img = Image.fromarray(np.uint8(img))

        # magical RoI borrowed from the official code
        # of IODINE
        img = img.crop([64, 29, 256, 221])
        return self.transform(img)

    def load_segs(self, meta_path):
        img = np.load(meta_path)['mask'][0, 0]
        img = 255 - np.dstack([img, img, img])
        img = Image.fromarray(np.uint8(img)).convert('1')

        img = img.crop([64, 29, 256, 221])
        return self.transform_seg(img)

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


def get_imgs(img_path, imsize, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    cimg = img
    if transform is not None:
        cimg = transform(cimg)

    retf = []
    retc = []
    # re_cimg = transforms.Resize(imsize[1])(cimg)
    re_cimg = transforms.Resize([128, 128])(cimg)
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
        #self.bbox_files = {}
        #self.bbox = self.load_bbox()
        self.filenames = make_dataset(data_dir)
        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    # only used in background stage

    def load_bbox(self):
        # Returns a dictionary with image filename as 'key' and its bounding box coordinates as 'value'

        data_dir = self.data_dir
        files_path = os.path.join(data_dir, 'Annotation')
        for target in os.listdir(files_path):
            d = os.path.join(files_path, target)
            for target in os.listdir(d):
                #print((len(os.listdir(d))))
                d_bbox = os.path.join(d, target)
                files = open(d_bbox)
                d_type = os.path.join(d_bbox.split('/', -1)[-2], d_bbox.split('/', -1)[-1])
                #print(d_type)
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

                #print(self.bbox_files.keys().__len__())  20580
                self.bbox_files.update(files_all)


        return self.bbox_files


    def prepair_training_pairs(self, index):

        key_files = self.filenames[index].split("/", -1)[-1].split('.')[0]
        key = os.path.join(self.filenames[index].split("/", -1)[-2], key_files)

        data_dir = self.data_dir
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
        img_name = '%s/jpg/%s.jpg' % (data_dir, key)
        _, imgs, _ = get_imgs(img_name, self.imsize, self.transform, normalize=self.norm)

        return imgs, c_code, key

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)


def get_dataloader(bs=None):
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))

    image_transform = transforms.Compose([
                                          transforms.Resize(int(imsize * 76 / 64)),
                                          transforms.RandomCrop(imsize),
                                          transforms.RandomHorizontalFlip()])



    # no to tensor!

    dataset = Dataset_FineGAN(cfg.DATA_DIR, base_size=cfg.TREE.BASE_SIZE, transform=image_transform)

    if bs == None:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=True, shuffle=True)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=True, shuffle=True)

    return dataset, dataloader


class Dataset(data.Dataset):
    def __init__(self, data_dir,data_name_dir, mode='train'):

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
        self.img_paths, self.labels = self.load_filenames(data_name_dir, mode)

        if mode == 'train':
            self.iterator = self.prepare_training_pairs
        else:
            self.iterator = self.prepare_test_pairs

    def load_filenames(self, data_name_dir, mode):
        if mode == 'train':
            with open(os.path.join(data_name_dir, 'trainset.txt'), 'r') as f:
                data = f.readlines()
                
        else:
            with open(os.path.join(data_name_dir, 'testset.txt'), 'r') as f:
                data = f.readlines()
        img_paths = [os.path.join(self.data_dir , 'jpg', _.split()[0].split('/')[-1]) for _ in data]
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
