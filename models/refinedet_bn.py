import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from data import voc_refinedet, coco_refinedet
import os

import numpy as np
from itertools import product as product
from functools import partial
from six.moves import map, zip
from math import sqrt as sqrt
try:
    # mmd
    from mmcv.ops import DeformConv2d
except:
    # solo
    from mmdet.ops import DeformConv as DeformConv2d
from mmcv.cnn import normal_init, kaiming_init, constant_init, xavier_init

class RefineDet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, ARM, ADM, TCB, num_classes, bn=True):
        super(RefineDet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco_refinedet, voc_refinedet)[num_classes == 21][str(size)]
        self.size = size
        self.bn = bn
        if size != 512 and size != 320:
            self.conv3_3_layer = (16, 23)[self.bn]
        self.conv4_3_layer = (23, 33)[self.bn]
        self.conv5_3_layer = (30, 43)[self.bn]
        self.extra_1_layer = (4, 6)[self.bn]
        if size == 640 or size == 5126 or size == 768:
            self.extra_2_layer = (8, 12)[self.bn]

        # for calc offset of ADM
        self.variance = self.cfg['variance']
        self.aspect_ratio = 2
        self.anchor_stride_ratio = 4
        self.anchor_num = 3
        self.dcn_kernel = 3
        self.dcn_pad = 1
        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        if size != 512 and size != 320:
            self.conv3_3_L2Norm = L2Norm(256, 10)
        self.conv4_3_L2Norm = L2Norm(512, 10)
        self.conv5_3_L2Norm = L2Norm(512, 8)
        self.extras = nn.ModuleList(extras)

        self.arm_loc = nn.ModuleList(ARM[0])
        self.arm_conf = nn.ModuleList(ARM[1])
        self.adm_loc1 = nn.ModuleList(ADM[0][0])
        self.adm_loc2 = nn.ModuleList(ADM[0][1])
        self.adm_loc3 = nn.ModuleList(ADM[0][2])
        self.adm_conf1 = nn.ModuleList(ADM[1][0])
        self.adm_conf2 = nn.ModuleList(ADM[1][1])
        self.adm_conf3 = nn.ModuleList(ADM[1][2])

        #self.tcb = nn.ModuleList(TCB)
        self.tcb0 = nn.ModuleList(TCB[0])
        self.tcb1 = nn.ModuleList(TCB[1])
        self.tcb2 = nn.ModuleList(TCB[2])
        self.step = len(self.cfg['feature_maps']) - 1

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        tcb_source = list()
        arm_loc = list()
        arm_conf = list()
        adm_loc = list()
        adm_conf = list()
        if self.phase == 'test':
            feat_sizes = list()

        # apply vgg up to conv4_3 relu and conv5_3 relu
        for k in range(self.conv5_3_layer):
            x = self.vgg[k](x)
            if self.size != 512 and self.size != 320 and self.conv3_3_layer - 1 == k:
                s = self.conv3_3_L2Norm(x)
                sources.append(s)
                if self.phase == 'test':
                    feat_sizes.append(x.shape[2:])
            if self.conv4_3_layer - 1 == k:
                s = self.conv4_3_L2Norm(x)
                sources.append(s)
                if self.phase == 'test':
                    feat_sizes.append(x.shape[2:])
            elif self.conv5_3_layer - 1 == k:
                s = self.conv5_3_L2Norm(x)
                sources.append(s)
                if self.phase == 'test':
                    feat_sizes.append(x.shape[2:])

        # apply vgg up to fc7
        for k in range(self.conv5_3_layer, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        if self.phase == 'test':
            feat_sizes.append(x.shape[2:])

        # apply extra layers and cache source layer outputs
        for k in range(len(self.extras)):
            x = self.extras[k](x)
            if self.extra_1_layer - 1 == k:
                sources.append(x)
                if self.phase == 'test':
                    feat_sizes.append(x.shape[2:])
            if (self.size == 640 or self.size == 5126 or self.size == 768) and self.extra_2_layer - 1 == k:
                sources.append(x)
                if self.phase == 'test':
                    feat_sizes.append(x.shape[2:])

        # apply ARM and ADM to source layers
        arm_loc_align = list()
        for (x, l, c) in zip(sources, self.arm_loc, self.arm_conf):
            loc_tensor = l(x)
            arm_loc_align.append(loc_tensor.detach())
            # arm_loc_align.append(loc_tensor)
            arm_loc.append(loc_tensor.permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        # calculate init ponits of offset before shape change
        adm_points = self.get_ponits(arm_loc_align)
        
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)

        # calculate TCB features
        p = None
        for k, v in enumerate(sources[::-1]):
            s = v
            for i in range(3):
                s = self.tcb0[(self.step-k)*3 + i](s)
            if k != 0:
                u = p
                u = self.tcb1[self.step-k](u)
                s += u
            for i in range(3):
                s = self.tcb2[(self.step-k)*3 + i](s)
            p = s
            tcb_source.append(s)
        tcb_source.reverse()

        # apply alignconv to source layers
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        for (x, ponits, l1, l2, l3, c1, c2, c3) in zip(
            tcb_source, adm_points, 
            self.adm_loc1, self.adm_loc2, self.adm_loc3, 
            self.adm_conf1, self.adm_conf2, self.adm_conf3
            ):
            loc = []
            conf = []
            dcn_offset1 = ponits[:, 0, ...].contiguous() - dcn_base_offset
            dcn_offset2 = ponits[:, 1, ...].contiguous() - dcn_base_offset
            dcn_offset3 = ponits[:, 2, ...].contiguous() - dcn_base_offset
            loc.append(l1(x, dcn_offset1))
            loc.append(l2(x, dcn_offset2))
            loc.append(l3(x, dcn_offset3))
            conf.append(c1(x, dcn_offset1))
            conf.append(c2(x, dcn_offset2))
            conf.append(c3(x, dcn_offset3))
            adm_loc.append(torch.cat(loc, 1).permute(0, 2, 3, 1).contiguous())
            adm_conf.append(torch.cat(conf, 1).permute(0, 2, 3, 1).contiguous())
        adm_loc = torch.cat([o.view(o.size(0), -1) for o in adm_loc], 1)
        adm_conf = torch.cat([o.view(o.size(0), -1) for o in adm_conf], 1)

        if self.phase == "test":
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),           # arm loc preds
                self.softmax(arm_conf.view(arm_conf.size(0), -1,
                             2)),                               # arm conf preds
                adm_loc.view(adm_loc.size(0), -1, 4),           # adm loc preds
                self.softmax(adm_conf.view(adm_conf.size(0), -1,
                             self.num_classes)),                # adm conf preds
                feat_sizes
            )
        else:
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 2),
                adm_loc.view(adm_loc.size(0), -1, 4),
                adm_conf.view(adm_conf.size(0), -1, self.num_classes),
            )
        return output

    def get_ponits(self, arm_loc):
        return multi_apply(self.get_ponits_single, arm_loc)

    # modified from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/reppoints_head.py
    def get_ponits_single(self, reg):
        scale = self.anchor_stride_ratio / 2
        anchors = [-scale, -scale, scale, scale]
        ls = scale*sqrt(self.aspect_ratio)
        ss = scale/sqrt(self.aspect_ratio)
        anchors += [-ls, -ss, ls, ss]
        anchors += [-ss, -ls, ss, ls]
        previous_boxes = reg.new_tensor(anchors).view(1, 3, 4, 1, 1)

        b, _, h, w = reg.shape
        reg = reg.view(b, self.anchor_num, 4, h, w)

        bxy = (previous_boxes[:, :, :2, ...] + previous_boxes[:, :, 2:, ...]) / 2.
        bwh = (previous_boxes[:, :, 2:, ...] -
               previous_boxes[:, :, :2, ...]).clamp(min=1e-6)
        grid_topleft = bxy + bwh * reg[:, :, :2, ...] * self.variance[0] - 0.5 * bwh * torch.exp(
            reg[:, :, 2:, ...]) * self.variance[1]
        grid_wh = bwh * torch.exp(reg[:, :, 2:, ...]) * self.variance[1]
        grid_left = grid_topleft[:, :, [0], ...]
        grid_top = grid_topleft[:, :, [1], ...]
        grid_width = grid_wh[:, :, [0], ...]
        grid_height = grid_wh[:, :, [1], ...]
        intervel = torch.tensor([(2 * i - 1) / (2 * self.dcn_kernel) for i in range(1, self.dcn_kernel + 1)]).view(
            1, 1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, self.anchor_num, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(3).repeat(1, 1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, self.anchor_num, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=3)
        grid_yx = grid_yx.view(b, self.anchor_num, -1, h, w)
    
        return grid_yx
    
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            vgg_weights = torch.load(pretrained)
            print('Loading base network...')
            self.vgg.load_state_dict(vgg_weights)
        elif pretrained is None:
            for m in self.vgg.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')
        # initialize newly added layers' weights with xavier method
        self.extras.apply(init_method)
        self.arm_loc.apply(init_method)
        self.arm_conf.apply(init_method)
        self.tcb0.apply(init_method)
        self.tcb1.apply(init_method)
        self.tcb2.apply(init_method)
        # initialize deform conv layers with normal method
        for adm_loc in (self.adm_loc1, self.adm_loc2, self.adm_loc3):
            for m in adm_loc:
                normal_init(m, std=0.01)
        for adm_conf in (self.adm_conf1, self.adm_conf2, self.adm_conf3):
            for m in adm_conf:
                normal_init(m, std=0.01)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def init_method(m):
    if isinstance(m, nn.Conv2d):
        xavier_init(m, distribution='uniform', bias=0)
    elif isinstance(m, nn.ConvTranspose2d):
        xavier_init(m, distribution='uniform', bias=0)
    elif isinstance(m, nn.BatchNorm2d):
        constant_init(m, 1)

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map_results)

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    if batch_norm:
        layers += [pool5, conv6, nn.BatchNorm2d(conv6.out_channels),
                   nn.ReLU(inplace=True), conv7, nn.BatchNorm2d(conv7.out_channels), nn.ReLU(inplace=True)]
    else:
        layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                conv2d = nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[k + 1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v
    return layers

def arm_multibox(in_channels, anchor_nums):
    arm_loc_layers = []
    arm_conf_layers = []
    for in_channel, anchor_num in zip(in_channels, anchor_nums):
        arm_loc_layers += [nn.Conv2d(in_channel, anchor_num * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(in_channel, anchor_num * 2, kernel_size=3, padding=1)]
    return (arm_loc_layers, arm_conf_layers)

def adm_multibox(level_channels, anchor_nums, num_classes):
    assert set(anchor_nums) == {3}
    adm_loc_layers1 = []
    adm_loc_layers2 = []
    adm_loc_layers3 = []
    adm_conf_layers1 = []
    adm_conf_layers2 = []
    adm_conf_layers3 = []
    for _ in level_channels:
            adm_loc_layers1 += [DeformConv2d(256, 4, kernel_size=3, padding=1)]
            adm_loc_layers2 += [DeformConv2d(256, 4, kernel_size=3, padding=1)]
            adm_loc_layers3 += [DeformConv2d(256, 4, kernel_size=3, padding=1)]
            adm_conf_layers1 += [DeformConv2d(256, num_classes, kernel_size=3, padding=1)]
            adm_conf_layers2 += [DeformConv2d(256, num_classes, kernel_size=3, padding=1)]
            adm_conf_layers3 += [DeformConv2d(256, num_classes, kernel_size=3, padding=1)]
    return (
        (adm_loc_layers1, adm_loc_layers2, adm_loc_layers3), 
        (adm_conf_layers1, adm_conf_layers2, adm_conf_layers3))

def odm_multibox(level_channels, anchor_nums, num_classes):
    odm_loc_layers = []
    odm_conf_layers = []
    for i in range(len(level_channels)):
        odm_loc_layers += [nn.Conv2d(256, anchor_nums[i] * 4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, anchor_nums[i] * num_classes, kernel_size=3, padding=1)]
    return (odm_loc_layers, odm_conf_layers)

def add_tcb(cfg):
    feature_scale_layers = []
    feature_upsample_layers = []
    feature_pred_layers = []
    for k, v in enumerate(cfg):
        feature_scale_layers += [nn.Conv2d(cfg[k], 256, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, 3, padding=1)
        ]
        feature_pred_layers += [nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True)
        ]
        if k != len(cfg) - 1:
            feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]
    return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)

base = {
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '5125': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '5126': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '640': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '768': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '320': [256, 'S', 512],
    '512': [256, 'S', 512],
    '5125': [256, 'S', 512],
    '5126': [256, 'S', 512, 128, 'S', 256],
    '640': [256, 'S', 512, 128, 'S', 256],
    '768': [256, 'S', 512, 128, 'S', 256],
}
mbox = {
    '320': [3, 3, 3, 3],  # number of boxes per feature map location
    '512': [3, 3, 3, 3],  # number of boxes per feature map location
    '5125': [3, 3, 3, 3, 3],  # number of boxes per feature map location
    '5126': [3, 3, 3, 3, 3, 3],  # number of boxes per feature map location
    '640': [3, 3, 3, 3, 3, 3],  # number of boxes per feature map location
    '768': [3, 3, 3, 3, 3, 3],  # number of boxes per feature map location
}

tcb = {
    '320': [512, 512, 1024, 512],
    '512': [512, 512, 1024, 512],
    '5125': [256, 512, 512, 1024, 512],
    '5126': [256, 512, 512, 1024, 512, 256],
    '640': [256, 512, 512, 1024, 512, 256],
    '768': [256, 512, 512, 1024, 512, 256],
}

arm = {
    '512': [512, 512, 1024, 512],
    '5125': [256, 512, 512, 1024, 512],
    '5126': [256, 512, 512, 1024, 512, 256],
    '640': [256, 512, 512, 1024, 512, 256],
    '768': [256, 512, 512, 1024, 512, 256],
}

def build_refinedet(phase, size=320, num_classes=21, backbone_dict=dict(bn=True)):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    bn = backbone_dict['bn']
    base_ = vgg(base[str(size)], 3, bn)
    extras_ = add_extras(extras[str(size)], 1024, bn)
    ARM_ = arm_multibox(arm[str(size)], mbox[str(size)])
    ADM_ = adm_multibox(arm[str(size)], mbox[str(size)], num_classes)
    TCB_ = add_tcb(tcb[str(size)])
    return RefineDet(phase, size, base_, extras_, ARM_, ADM_, TCB_, num_classes, bn)
