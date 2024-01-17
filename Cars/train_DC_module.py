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
from model import G_NET_DCS, D_NET_DCS
from datasets_cars import *
import random
from glob import glob
from inception_score import inception_score
import copy
from torchvision.transforms.functional import to_tensor
from utils import *
from itertools import chain
from copy import deepcopy
from tensorboardX import summary
from tensorboardX import FileWriter
import torchvision.transforms as transforms
from inception import InceptionV3
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

def define_optimizers(netG, netD):
    optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    return optimizerD, optimizerG

def load_network():
    # prepare G net
    netG = G_NET_DCS(cfg.FINE_GRAINED_CATEGORIES, fore_num=3, mask_num=1, back_num=1).to(device)
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)

    netD = D_NET_DCS(cfg.FINE_GRAINED_CATEGORIES)
    netD.apply(weights_init)
    netD = torch.nn.DataParallel(netD, device_ids=gpus)

    return netG, netD

def save_model(netG, netD, epoch, model_dir):
    torch.save(netG.state_dict(), '%s/G_%d.pth' % (model_dir, epoch))
    torch.save(netD.state_dict(), '%s/D3_%d.pth' % (model_dir, epoch))

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
        self.test_dataset = Dataset(data_dir=cfg.DATA_DIR, data_name_dir=cfg.DATA_DIR, mode='test')
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=False, num_workers=8, shuffle=False)

        self.netG, self.netD = load_network()
        self.optimizerD, self.optimizerG = define_optimizers(self.netG, self.netD)
        self.max_epoch = cfg.TRAIN.FIRST_MAX_EPOCH
        self.CE = nn.CrossEntropyLoss()
        self.RF_loss = nn.BCELoss()

        print('Get the statistic of training images for computing fid score.')
        self.inception = InceptionV3([3]).to(device)
        self.inception.eval()
        pred_arr = np.empty((len(self.dataset), 2048))
        start_idx = 0
        for data in self.dataloader:
            batch = data[1].to(device)
            with torch.no_grad():
                pred = self.inception(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = nn.AdaptiveAvgPool2d(pred, output_size=(1, 1))
                # adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]
        self.mu = np.mean(pred_arr, axis=0)
        self.sig = np.cov(pred_arr, rowvar=False)

        if not cfg.TRAIN.FLAG:
            G = "../saved_models/cars/netG.pth"
            D = "../saved_models/cars/netD.pth"
            self.eval(path_G=G, path_D=D)
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

    def eval(self, path_G, path_D):
        netG = load_model(self.netG, path_G)
        netD = load_model(self.netD, path_D)
        BATCH = int(20000 / self.num_cls)
        for temp in range(0, self.num_cls):
            c_code = torch.zeros([BATCH, cfg.FINE_GRAINED_CATEGORIES])
            c_code[:, temp] = 1
            b_code = c_code
            noise = torch.FloatTensor(BATCH, cfg.GAN.Z_DIM).normal_(0, 1).to(device) * 4
            _, fake_final_img\
                = netG(noise, c_code, b_code)
            for i in range(0, BATCH):
                save_img_results(None, fake_final_img[i].detach().cpu(), temp, i, self.image_dir, flage=0)
        ## get inception score on 20,000 samples
        print('20,000 images have been generated!')
        dataset = ImageDataset(self.image_dir, exts=['png', 'jpg'])
        print(inception_score(dataset, cuda=True, batch_size=32, resize=True, splits=10))

        ## get acc and nmi scores on predictions
        pred_c = []; real_c = []
        with torch.no_grad():
            for img, lab in self.test_dataloader:
                real_img = img.to(device)
                feat, _, _, class_emb = netD(real_img, self.eye)
                f = F.normalize(feat, p=2, dim=1)
                c = F.normalize(class_emb, p=2, dim=1)
                class_dist = torch.cat(
                    [torch.matmul(f, c[i]).unsqueeze(-1) / self.temp for i in range(self.num_cls)], 1)
                pred_c += list(torch.argmax(class_dist, 1).cpu().numpy())
                real_c += list(lab.cpu().numpy())
        c_table = get_stat(pred_c, self.num_cls, real_c)
        for i in range(1, self.over):
            c_table[self.num_cls // self.over * i:self.num_cls // self.over * (i + 1), :] = \
                c_table[:self.num_cls // self.over, :]
        idx_map = get_match(c_table)
        cur_acc = get_acc(c_table, idx_map, self.over)
        cur_nmi = get_nmi(c_table[:self.num_gt_cls, :])
        ## get fid score on randomly generated samples
        pred_arr = np.empty((len(self.dataset), 2048))
        start_idx = 0
        for i in range(len(self.dataset) // self.batch_size):
            real_z, real_c, real_c_inv = self.prepare_code()
            with torch.no_grad():
                _,  fake_final_img = netG(real_z, real_c, real_c_inv)
                pred = self.inception(fake_final_img)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                print('size mismatch error occurred during the fid score computation!')
                pred = nn.AdaptiveAvgPool2d(pred, output_size=(1, 1))
                # adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]
        cur_mu = np.mean(pred_arr, axis=0)
        cur_sig = np.cov(pred_arr, rowvar=False)
        cur_fid = calculate_frechet_distance(self.mu, self.sig, cur_mu, cur_sig)
        print("eval finished", "\t fid : ", "{:.3f}".format(cur_fid), \
              "\t acc : ", "{:.3f}".format(cur_acc), \
              "\t nmi : ", "{:.3f}".format(cur_nmi))

    def train(self):
        print('===> Start Training.')
        for self.epoch in range(cfg.TRAIN.FIRST_MAX_EPOCH):
            self.count = 0
            for data in self.dataloader:
                self.count = self.count + 1
                self.aug_img, self.real_img, self.real_z, self.real_c, self.real_b = self.prepare_data(data)

                self.mask, self.fake_final_img \
                    = self.netG(self.real_z, self.real_c, self.real_b)

                # synthesize images with random back_code
                rand_b = torch.zeros(self.batch_size, cfg.SUPER_CATEGORIES).to(device)
                rand_idx = [i for i in range(cfg.SUPER_CATEGORIES)]
                random.shuffle(rand_idx)
                with torch.no_grad():
                    for i, idx in enumerate(rand_idx[:self.batch_size]):
                        rand_b[i, idx] = 1
                    _,  self.fake_final_img_rand \
                        = self.netG(self.real_z, self.real_c, rand_b)

                # train D and G
                self.train_D()
                self.train_G()

            save_img_results(self.fake_final_img.detach().cpu(), None, self.count, self.epoch, self.image_dir, flage=1)
            save_img_results(None, self.mask.detach().cpu(), self.count, self.epoch, self.image_dir, flage=1)

            if self.epoch % 10 == 0:
                self.netG.eval()
                self.netD.eval()
                save_model(self.netG,  self.netD, self.epoch, self.model_dir)
                ## get acc and nmi scores on predictions
                pred_c = []
                real_c = []
                with torch.no_grad():
                    for img, lab in self.test_dataloader:
                        real_img = img.to(device)
                        feat, _, _, class_emb = self.netD(real_img, self.eye)
                        f = F.normalize(feat, p=2, dim=1)
                        c = F.normalize(class_emb, p=2, dim=1)
                        class_dist = torch.cat(
                            [torch.matmul(f, c[i]).unsqueeze(-1) / self.temp for i in range(self.num_cls)], 1)
                        pred_c += list(torch.argmax(class_dist, 1).cpu().numpy())
                        real_c += list(lab.cpu().numpy())

                c_table = get_stat(pred_c, self.num_cls, real_c)
                for i in range(1, self.over):
                    c_table[self.num_cls // self.over * i:self.num_cls // self.over * (i + 1), :] = \
                        c_table[ :self.num_cls // self.over, :]
                idx_map = get_match(c_table)
                cur_acc = get_acc(c_table, idx_map, self.over)
                cur_nmi = get_nmi(c_table[:self.num_gt_cls, :])
                # get fid score on randomly generated samples
                pred_arr = np.empty(((len(self.dataset) // self.batch_size) * self.batch_size, 2048))
                start_idx = 0
                for i in range(len(self.dataset) // self.batch_size):
                    real_z, real_c, real_c_inv = self.prepare_code()
                    with torch.no_grad():
                        _, fake_final_img = self.netG(real_z, real_c, real_c_inv)
                        pred = self.inception(fake_final_img)[0]
                    if pred.size(2) != 1 or pred.size(3) != 1:
                        print('size mismatch error occurred during the fid score computation!')
                        pred = nn.AdaptiveAvgPool2d(pred, output_size=(1, 1))
                        # adaptive_avg_pool2d(pred, output_size=(1, 1))
                    pred = pred.squeeze(3).squeeze(2).cpu().numpy()
                    pred_arr[start_idx:start_idx + pred.shape[0]] = pred
                    start_idx = start_idx + pred.shape[0]
                cur_mu = np.mean(pred_arr, axis=0)
                cur_sig = np.cov(pred_arr, rowvar=False)
                cur_fid = calculate_frechet_distance(self.mu, self.sig, cur_mu, cur_sig)
                print(str(self.epoch) + "th epoch finished", "\t fid : ", "{:.3f}".format(cur_fid), \
                      "\t acc : ", "{:.3f}".format(cur_acc), \
                      "\t nmi : ", "{:.3f}".format(cur_nmi))

                self.netG.train()
                self.netD.train()


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

    # evual  18.57 acc :  0.158 	 nmi :  0.430
    # (2.733426460383631, 0.02290502933475585)