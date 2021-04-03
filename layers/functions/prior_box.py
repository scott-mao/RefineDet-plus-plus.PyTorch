from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg, box_dimension=None, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        if phase == 'train':
            self.image_size = (cfg['min_dim'], cfg['min_dim'])
            self.feature_maps = list(zip(cfg['feature_maps'], cfg['feature_maps']))
        elif phase == 'test':
            self.image_size = image_size
            self.feature_maps = box_dimension
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f[0]), range(f[1])):
                f_kx = self.image_size[1] / self.steps[k]
                f_ky = self.image_size[0] / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_kx
                cy = (i + 0.5) / f_ky

                # aspect_ratio: 1
                # rel size: min_size
                s_kx = self.min_sizes[k]/self.image_size[1]
                s_ky = self.min_sizes[k]/self.image_size[0]
                mean += [cx, cy, s_kx, s_ky]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_kx*sqrt(ar), s_ky/sqrt(ar)]
                    mean += [cx, cy, s_kx/sqrt(ar), s_ky*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
