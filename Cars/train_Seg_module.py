##############
# This is first going
# work hard!!
# best wish
#################

from config import cfg
import os
import time
from PIL import Image
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from model import G_NET_DCS
from datasets_cars import *
import random
from glob import glob
from inception_score import inception_score
import copy
from torchvision.transforms.functional import to_tensor
from Unet import U_Net
from itertools import chain
from copy import deepcopy
from tensorboardX import summary
from tensorboardX import FileWriter
import torchvision.transforms as transforms
from inception import InceptionV3
from utils import *
from evals import *
cudnn.benchmark = True

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cfg.GPU_ID = "0"
device = torch.device("cuda:" + cfg.GPU_ID)
gpus = [int(ix) for ix in cfg.GPU_ID.split(',')]

class ImageDataset(Dataset):
    def __init__(self, path, exts=['png', 'jpg']):
        self.paths = []
        for ext in exts:
            self.paths.extend(
                list(glob(os.path.join(path, '*.%s' % ext))))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        tensor = copy.deepcopy(to_tensor(image))
        image.close()
        return tensor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def load_model(model, model_path):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    toggle_grad(model, False)
    return model.eval()

def define_optimizers(netU):
    optimizerU = optim.Adam(netU.parameters(), lr=2e-4, betas=(0.5, 0.999))
    return optimizerU

def evaluate(netEncM, loader, device):
    nbIter = 0
    iou_s = 0
    dice_s = 0

    for x_data in loader:
        xLoad = x_data[0]
        mLoad = x_data[1]
        # for xLoad, mLoad in loader:
        xData = xLoad.to(device)
        mData = mLoad.to(device)
        mPred = netEncM(xData)
        nbIter += 1

        bs = xData.size()[0]
        pred_f = mPred >= 0.5
        pred_b = mPred < 0.5
        gt = mData

        iou = torch.max(
            (pred_f * gt).view(bs, -1).sum(dim=-1) / \
            ((pred_f + gt) >= 1).view(bs, -1).sum(dim=-1),
            (pred_b * gt).view(bs, -1).sum(dim=-1) / \
            ((pred_b + gt) >= 1).view(bs, -1).sum(dim=-1))

        dice = torch.max(
            2 * (pred_f * gt).view(bs, -1).sum(dim=-1) / \
            (pred_f.view(bs, -1).sum(dim=-1) + gt.view(bs, -1).sum(dim=-1)),
            2 * (pred_b * gt).view(bs, -1).sum(dim=-1) / \
            (pred_b.view(bs, -1).sum(dim=-1) + gt.view(bs, -1).sum(dim=-1)))
        iou_s += iou.mean().item()
        dice_s += dice.mean().item()

    return iou_s / nbIter, dice_s / nbIter

def load_network():
    # prepare G net
    netG = G_NET_DCS(cfg.FINE_GRAINED_CATEGORIES, fore_num=3, mask_num=1, back_num=1).to(device)
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)

    netU = U_Net()
    netU.apply(weights_init)
    netU = torch.nn.DataParallel(netU, device_ids=gpus)

    return netG, netU

def save_model(netG, netD, epoch, model_dir):
    torch.save(netG.state_dict(), '%s/G_%d.pth' % (model_dir, epoch))
    torch.save(netD.state_dict(), '%s/D3_%d.pth' % (model_dir, epoch))

def save_netU(netU, epoch, model_dir):
    torch.save(netU.state_dict(), '%s/U_%d.pth' % (model_dir, epoch))

def binary_entropy(syn_mask, pre_mask):
    return -syn_mask * torch.log2(pre_mask + 1e-6) - (1 - syn_mask) * torch.log2(1 - pre_mask + 1e-6)


class BinaryLoss(nn.Module):

    def __init__(self, loss_weight):
        super(BinaryLoss, self).__init__()
        self.loss_weight = loss_weight

    @staticmethod
    def binary_entropy(p):
        return -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)

    def __call__(self, mask):
        return self.loss_weight * self.binary_entropy(mask).mean()

class Trainer(object):
    def __init__(self, output_dir):
        # make dir for all kinds of output

        self.model_dir = os.path.join(output_dir, 'Model')
        os.makedirs(self.model_dir)
        self.image_dir = os.path.join(output_dir, 'Image')
        os.makedirs(self.image_dir)
        self.opt_dir = os.path.join(output_dir, 'Opt')
        os.makedirs(self.opt_dir)

        # other variables
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.over = cfg.OVER
        self.num_cls = cfg.FINE_GRAINED_CATEGORIES
        self.num_gt_cls = int(self.num_cls/self.over)
        self.temp = cfg.TEMP
        self.eye = torch.eye(self.num_cls).to(device)

        self.real_labels = torch.ones_like(torch.randn(self.batch_size, 1).to(device))
        self.fake_labels = torch.zeros_like(torch.randn(self.batch_size, 1)).to(device)

        # make dataloader and code buffer
        self.dataset, self.dataloader = get_dataloader()
        self.netG, self.netU = load_network()
        self.optimizerU = define_optimizers(self.netU)
        self.max_epoch = cfg.TRAIN.FIRST_MAX_EPOCH

        if cfg.TRAIN.FLAG:
            U = "../saved_models/cars/netU.pth"
            netU = load_model(self.netU, U)
            config = Config('./car_all.yml')
            trainset = CarDataset(config, data_split=config.TEST_SPLIT, use_flip=False)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)
            iou_s, dice_s = evaluate(netU, trainloader, device)
            print("eval finished", "\t iou_s : ", "{:.3f}".format(iou_s), \
                  "\t dice_s : ", "{:.3f}".format(dice_s))
            print('===> Evaluation Completed ')
### ===> Evaluation Completed

    def prepare_data(self, data):
        aug_img, real_img, real_c, _, _ = data

        real_img = real_img.to(device)
        aug_img = aug_img.to(device)
        real_z = torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM).normal_(0, 1).to(device) * 4
        real_c = real_c.to(device)
        real_b = child_to_parent(real_c, cfg.FINE_GRAINED_CATEGORIES, cfg.SUPER_CATEGORIES).to(device)

        return aug_img, real_img, real_z, real_c, real_b

    def prepare_code(self):
        rand_z = torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM).normal_(0, 1).to(device) * 4
        rand_c = torch.zeros(self.batch_size, self.num_cls).to(device)

        rand_idx = [i for i in range(self.num_cls)]
        random.shuffle(rand_idx)
        for i, idx in enumerate(rand_idx[:self.batch_size]):
            rand_c[i, idx] = 1
        rand_b = child_to_parent(rand_c, cfg.FINE_GRAINED_CATEGORIES, cfg.SUPER_CATEGORIES).to(device)

        return rand_z, rand_c, rand_b

    def train(self):
        print('===> Start Training.')
        for self.epoch in range(cfg.TRAIN.FIRST_MAX_EPOCH):
            self.count = 0
            for data in self.dataloader:
                self.count = self.count + 1
                self.aug_img, self.real_img, self.real_z, self.real_c, self.real_b = self.prepare_data(data)
                with torch.no_grad():
                    self.mask, self.fake_final_img \
                        = self.netG(self.real_z, self.real_c, self.real_b)

                # train U
                self.train_U()
            save_img_results(None, self.mask_fake.detach().cpu(), self.count, self.epoch, self.image_dir, flage=1)

            # ==> evaluate
            self.netU.eval()

            save_netU(self.netU, self.epoch, self.model_dir)
            config = Config('./car_all.yml')
            trainset = CarDataset(config, data_split=config.TEST_SPLIT, use_flip=False)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)
            iou_s, dice_s = evaluate(self.netU, trainloader, device)
            print(str(self.epoch) + "th epoch finished", "\t iou_s : ", "{:.3f}".format(iou_s), \
                  "\t dice_s : ", "{:.3f}".format(dice_s))

            self.netU.train()


if __name__ == "__main__":
    manualSeed = 0
    print(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    output_dir = make_output_dir()
    trainer = Trainer(output_dir)
    print('start training now')
    trainer.train()

