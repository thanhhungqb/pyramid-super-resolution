"""
Implement multi level (pyramid) with super-resolution
"""
from fastai.vision import *

from outside.stn import STN
from outside.super_resolution.srnet import SRNet3
from prlab.fastai.utils import base_arch_str_to_obj
from prlab.torch.functions import PassThrough


class PyramidSR(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.multiples = config.get('multiples', [1, 2])

        # stn layer
        self.stn = STN(img_size=config['img_size'])

        # latest layer, classifier, main and shared layer
        base_arch = base_arch_str_to_obj(config.get('base_arch', 'vgg16_bn'))
        self.cl = create_cnn_model(base_arch=base_arch, nc=config['n_classes'])

        # some parallel branches, in different size with SR
        pyramid_group = []
        for mul in self.multiples:
            if mul == 1:
                # do nothing
                pyramid_group.append(PassThrough())
            else:
                pyramid_group.append(SRNet3(mul))

        self.pyramid_group = nn.Sequential(*pyramid_group)

    def layer_groups(self):
        return [
            self.stn,
            self.pyramid_group,
            self.cl
        ]

    def load_weights(self, **config):
        """
        Support to load weights from config.
        Name of key in config should be keep as general case:
            - base_weights_path for latest, cl layer
            - weight_path_x{n} for pyramid_group, except x1
        :param config:
        :return:
        """
        if config.get('base_weights_path', None) is not None:
            self.cl.load_state_dict(torch.load(config['base_weights_path']), strict=True)
            print('load weights for classifier', config['base_weights_path'])

        for idx in range(len(self.multiples)):
            xn_name = 'weight_path_x{}'.format(self.multiples[idx])
            if config.get(xn_name, None) is not None:
                out = self.pyramid_group[idx].load_state_dict(torch.load(config[xn_name]), strict=True)
                print('load weights for {}'.format(xn_name), out)

    def forward(self, *x, **kwargs):
        x = self.stn(x[0])

        branches_out = []
        for idx in range(len(self.pyramid_group)):
            o1 = self.pyramid_group[idx](x)
            x_out = self.cl(o1)
            branches_out.append(x_out)
        x_stack = torch.stack(branches_out, dim=-1)
        out = torch.mean(x_stack, dim=-1)
        return out
