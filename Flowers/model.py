import sys
import torch
import torch.nn as nn
import torch.nn.parallel
from config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Upsample
import time
from collections import deque

class GLU(nn.Module):

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

def conv3x3(in_planes, out_planes):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

def child_to_parent(child_c_code, classes_child, classes_parent):
    ratio = classes_child / classes_parent
    arg_parent = torch.argmax(child_c_code, dim=1) / ratio
    parent_c_code = torch.zeros([child_c_code.size(0), classes_parent]).cuda()
    for i in range(child_c_code.size(0)):
        parent_c_code[i][int(arg_parent[i])] = 1
    return parent_c_code

def upBlock(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                          conv3x3(in_planes, out_planes * 2),
                          nn.BatchNorm2d(out_planes * 2),
                          GLU())
    return block

def sameBlock(in_planes, out_planes):
    block = nn.Sequential(conv3x3(in_planes, out_planes * 2),
                          nn.BatchNorm2d(out_planes * 2),
                          GLU())
    return block

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(conv3x3(channel_num, channel_num * 2),
                                   nn.BatchNorm2d(channel_num * 2),
                                   GLU(),
                                   conv3x3(channel_num, channel_num),
                                   nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

# ============================= Generator ===================================
class GET_IMAGE(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.img = nn.Sequential(conv3x3(ngf, 3), nn.Tanh())

    def forward(self, h_code):
        return self.img(h_code)

class GET_MASK(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.img = nn.Sequential(conv3x3(ngf, 1), nn.Sigmoid())

    def forward(self, h_code):
        return self.img(h_code)

class FEATURE_DECODER_DCS(nn.Module):
    def __init__(self, ngf):
        super().__init__()

        self.ngf = ngf
        in_dim = cfg.GAN.Z_DIM

        self.fc = nn.Sequential(nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False), nn.BatchNorm1d(ngf * 4 * 4 * 2), GLU())
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)

    def forward(self, z_input):

        in_code = z_input
        out_code = self.fc(in_code).view(-1, self.ngf, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)

        return out_code

class SubNet_DCS_C(nn.Module):
    def __init__(self, ngf, num_residual=1):
        super().__init__()

        self.ngf = ngf
        self.code_len = 0
        self.num_residual = num_residual

        self.jointConv = sameBlock(self.code_len + self.ngf, ngf * 2)
        self.upsample4 = upBlock(ngf * 2, ngf )
        self.upsample5 = upBlock(ngf, ngf)
        self.residual = self._make_layer()
        self.downsample_1 = sameBlock(ngf, ngf // 4)

    def _make_layer(self):
        layers = []
        for _ in range(self.num_residual):
            layers.append(ResBlock(self.ngf))
        return nn.Sequential(*layers)

    def forward(self, h_code):
        h_c_code = h_code
        out_code = self.jointConv(h_c_code)
        out_code = self.upsample4(out_code)
        out_code = self.upsample5(out_code)
        out_code = self.residual(out_code)
        out_code = self.downsample_1(out_code)
        return out_code

class SubNet_DCS_B(nn.Module):
    def __init__(self, ngf, num_residual=1):
        super().__init__()

        self.ngf = ngf
        self.code_len = cfg.SUPER_CATEGORIES
        self.num_residual = num_residual

        self.jointConv = sameBlock(self.code_len + self.ngf, ngf * 2)
        self.upsample4 = upBlock(ngf * 2, ngf )
        self.upsample5 = upBlock(ngf, ngf)

        self.residual = self._make_layer()
        self.downsample_1 = sameBlock(ngf, ngf // 4)

    def _make_layer(self):
        layers = []
        for _ in range(self.num_residual):
            layers.append(ResBlock(self.ngf))
        return nn.Sequential(*layers)

    def forward(self, h_code, code):
        h, w = h_code.size(2), h_code.size(3)
        code = code.view(-1, self.code_len, 1, 1).repeat(1, 1, h, w)
        h_c_code = torch.cat((code, h_code), 1)

        out_code = self.jointConv(h_c_code)
        out_code = self.upsample4(out_code)
        out_code = self.upsample5(out_code)

        out_code = self.residual(out_code)
        out_code = self.downsample_1(out_code)
        return out_code

class G_NET_DCS(nn.Module):
    def __init__(self, class_code, fore_num, mask_num, back_num):
        super().__init__()
        ngf = cfg.GAN.GF_DIM

        self.parent_stage = FEATURE_DECODER_DCS(ngf * 8)
        # class code
        self.code_len = class_code
        self.jointConv = sameBlock(self.code_len + ngf, ngf)
        # mask networks
        self.second_stage = SubNet_DCS_C(ngf, num_residual=mask_num)
        self.second_mask = GET_MASK(ngf // 4)
        # fore networks
        self.colour_stage = SubNet_DCS_C(ngf, num_residual=fore_num)
        self.colour_image = GET_IMAGE(ngf // 4)
        # back networks
        self.colour_stag_inverse = SubNet_DCS_B(ngf, num_residual=back_num)
        self.colour_image_inverse = GET_IMAGE(ngf // 4)

    def forward(self, z, c_code, b_code):
        p_temp = self.parent_stage(z)
        # back network
        m_temp = self.colour_stag_inverse(p_temp, b_code)
        bg_img = self.colour_image_inverse(m_temp)
        # class code
        h, w = p_temp.size(2), p_temp.size(3)
        cz_code = c_code.view(-1, self.code_len, 1, 1).repeat(1, 1, h, w)
        p_temp = torch.cat((p_temp, cz_code), 1)
        p_temp = self.jointConv(p_temp)
        # fore and mask network
        m_temp = self.second_stage(p_temp)
        mask_img = self.second_mask(m_temp)
        m_temp = self.colour_stage(p_temp)
        fg_img = self.colour_image(m_temp)

        return mask_img, (fg_img * mask_img + (1. - mask_img) * bg_img)


class G_NET_DCS_syn(nn.Module):
    def __init__(self, class_code, fore_num, mask_num, back_num):
        super().__init__()
        ngf = cfg.GAN.GF_DIM

        self.parent_stage = FEATURE_DECODER_DCS(ngf * 8)
        # class code
        self.code_len = class_code
        self.jointConv = sameBlock(self.code_len + ngf, ngf)
        # mask networks
        self.second_stage = SubNet_DCS_C(ngf, num_residual=mask_num)
        self.second_mask = GET_MASK(ngf // 4)
        # fore networks
        self.colour_stage = SubNet_DCS_C(ngf, num_residual=fore_num)
        self.colour_image = GET_IMAGE(ngf // 4)
        # back networks
        self.colour_stag_inverse = SubNet_DCS_B(ngf, num_residual=back_num)
        self.colour_image_inverse = GET_IMAGE(ngf // 4)

    def forward(self, z, c_code, b_code):
        p_temp = self.parent_stage(z)
        # back network
        m_temp = self.colour_stag_inverse(p_temp, b_code)
        bg_img = self.colour_image_inverse(m_temp)
        # class code
        h, w = p_temp.size(2), p_temp.size(3)
        cz_code = c_code.view(-1, self.code_len, 1, 1).repeat(1, 1, h, w)
        p_temp = torch.cat((p_temp, cz_code), 1)
        p_temp = self.jointConv(p_temp)
        # fore and mask network
        m_temp = self.second_stage(p_temp)
        mask_img = self.second_mask(m_temp)
        m_temp = self.colour_stage(p_temp)
        fg_img = self.colour_image(m_temp)

        return mask_img, fg_img, bg_img, (fg_img * mask_img + (1. - mask_img) * bg_img)


# ==================== Discriminator ====================
def encode_img(ndf=64, in_c=3):
    layers = nn.Sequential(
        nn.Conv2d(in_c, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return layers

class D_NET_DCS(nn.Module):
    def __init__(self, c_dim, ndf=64):
        super(D_NET_DCS, self).__init__()
        self.ndf = ndf
        self.c_dim = c_dim
        self.base = encode_img()

        self.info_head = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(ndf * 8),
                                       nn.LeakyReLU(0.2, inplace=True),
                                       nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=4))


        self.cluster_projector = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
                                               nn.AdaptiveAvgPool2d((1, 1)))
        self.mlp = nn.Sequential(nn.Linear(ndf * 8, ndf * 8),
                                 nn.ReLU(),
                                 nn.Linear(ndf * 8, c_dim),
                                 nn.Softmax(dim=1))

        self.rf_head = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                                     nn.Sigmoid())

        self.centroids = nn.Linear(self.c_dim, ndf * 8)

    def forward(self, x, eye):
        out = self.base(x)
        cluster = self.cluster_projector(out).view(out.size()[0],-1)
        cluster = self.mlp(cluster)
        info = self.info_head(out).view(-1, self.ndf * 8)
        rf = self.rf_head(out).view(-1, 1)
        class_emb = self.centroids(eye)
        return info, cluster, rf, class_emb

