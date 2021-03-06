from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
from layers.box_utils import box_limits, center_size
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, config):
        super(PriorBox, self).__init__()
        self.image_size = config.input_size
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(config.prior_box_aspect_ratios)
        self.variance = config.prior_box_variance
        self.feature_maps_dim = config.feature_maps_dim
        self.scales = config.prior_box_scales
        self.aspect_ratios = config.prior_box_aspect_ratios
        self.clip = config.prior_box_clip
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

        # Compute box coordinates
        coordinates = []
        s = self.scales
        for k, f in enumerate(self.feature_maps_dim):
            for i, j in product(range(f), repeat=2):
                # unit center x,y
                cx = (j + 0.5) / f
                cy = (i + 0.5) / f

                # aspect_ratio: 1
                # rel size: min_size
                coordinates += [cx, cy, s[k], s[k]]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_prime = sqrt(s[k] * s[k+1])
                coordinates += [cx, cy, s_prime, s_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    coordinates += [cx, cy, s[k] * sqrt(ar), s[k]/sqrt(ar)]
        # back to torch land
        coordinates_tensor = torch.Tensor(coordinates).view(-1, 4)

        # clip prior boxes to fit the image
        if self.clip:
            coordinates_tensor = box_limits(coordinates_tensor)
            coordinates_tensor = coordinates_tensor.clamp_(min=0, max=1)
            coordinates_tensor = center_size(coordinates_tensor)
        self.coordinates = coordinates_tensor
