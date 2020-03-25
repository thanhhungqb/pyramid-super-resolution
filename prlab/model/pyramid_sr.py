"""
Implement multi level (pyramid) with super-resolution
"""
import deprecation
from fastai.vision import *

from outside.stn import STN
from outside.super_resolution.srnet import SRNet3
from prlab.fastai.utils import base_arch_str_to_obj
from prlab.gutils import load_func_by_name
from prlab.torch.functions import PassThrough


@deprecation.deprecated(
    details='replace by clean version at `prlab.model.pyramid_sr.PyramidSRNonShare`, keep this for reference only')
class PyramidSR(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.multiples = config.get('multiples', [1, 2])

        self.n_classes = config['n_classes']

        # stn layer
        self.stn = STN(img_size=config['img_size'])

        # latest layer, classifier, main and shared layer
        self.base_arch = base_arch_str_to_obj(config.get('base_arch', 'vgg16_bn'))
        self.cl = create_cnn_model(base_arch=self.base_arch, nc=config['n_classes'])

        # some parallel branches, in different size with SR
        pyramid_group = []
        for mul in self.multiples:
            # if mul == 1:
            #     # do nothing
            #     pyramid_group.append(PassThrough())
            # else:
            pyramid_group.append(self.make_layer_pyramid(mul))

        self.pyramid_group = nn.Sequential(*pyramid_group)

        # branch weighted
        self.n_branches = len(self.multiples)
        self.branch_weighted = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.n_branches * config['n_classes'], out_features=self.n_branches),
            nn.Softmax(dim=1)
        )
        # initial for all branches equals, meaning out will be [1/n_branches] * n_branches
        # set equals to all n_branches rows A of Ax+B
        nlayer = self.branch_weighted[-2]
        # just leave bias randomly
        torch.nn.init.normal_(nlayer.bias.data)

        xtmp = torch.normal(mean=0.0, std=0.2, size=(self.n_branches, self.n_branches * self.n_classes))
        for i in range(1, self.n_branches):
            xtmp[i, :] = xtmp[0, :]  # set all row equal, then Ax+B generate same output at the beginning

        nlayer.weight.data.copy_(xtmp)

        self.config = config
        self.is_testing = False

    def make_layer_pyramid(self, mul):
        """
        :param mul:
        :return:
        """
        if mul == 3:
            layer = nn.Sequential(
                SRNet3(mul),
                create_cnn_model(base_arch=self.base_arch, nc=self.n_classes)
            )
            layer[0].load_state_dict(torch.load('/ws/models/super_resolution/facial_x3.pth'), strict=True)
            layer[-1].load_state_dict(torch.load('/ws/models/ferplus/vgg16_bn_quick_3size_e20/final.w'), strict=True)
            return layer
        elif mul == 2:
            layer = nn.Sequential(
                SRNet3(mul),
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                create_cnn_model(base_arch=self.base_arch, nc=self.n_classes)
            )
            layer[0].load_state_dict(torch.load('/ws/models/super_resolution/facial_x2.pth'), strict=True)
            layer[-1].load_state_dict(torch.load('/ws/models/ferplus/vgg16_bn_quick_3size_e20/final.w'), strict=True)
            return layer
        else:
            layer = nn.Sequential(
                PassThrough(),
                create_cnn_model(base_arch=self.base_arch, nc=self.n_classes)
            )
            layer[-1].load_state_dict(torch.load('/ws/models/ferplus/vgg16_bn_quick_3size_e20/final.w'), strict=True)
            return layer

    def layer_groups(self):
        return [
            self.branch_weighted,
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
            # x_out = self.cl(o1)
            x_out = o1
            branches_out.append(x_out)
        x_stack = torch.stack(branches_out, dim=-1)  # [bs, n_classes, n_branches]
        weights = self.branch_weighted(x_stack.view([-1, self.n_classes * self.n_branches]))

        if not self.training and self.is_testing:
            # not valid, just test
            return weights_branches((x_stack, weights))

        return x_stack, weights


class PyramidSRNonShare(nn.Module):
    """
    Cleaning version for `prlab.model.pyramid_sr.PyramidSR`, which clear unused layers
    """

    def __init__(self, **config):
        super().__init__()
        self.multiples = config.get('multiples', [1, 2])
        self.config = config
        self.n_classes = config['n_classes']

        # stn layer
        self.stn = STN(img_size=config['img_size'])

        # latest layer, classifier, main and shared layer
        self.base_arch = base_arch_str_to_obj(config.get('base_arch', 'vgg16_bn'))

        # some parallel branches, in different size with SR
        pyramid_group = []
        for mul in self.multiples:
            pyramid_group.append(self.make_layer_pyramid(mul))

        self.pyramid_group = nn.Sequential(*pyramid_group)

        self.is_testing = False

    def make_layer_pyramid(self, mul):
        """
        :param mul:
        :return:
        """
        # TODO remove hard_code
        p_x2 = '/ws/models/super_resolution/facial_x2.pth'
        p_x3 = '/ws/models/super_resolution/facial_x3.pth'
        p_vgg = '/ws/models/ferplus/vgg16_bn_quick_3size_e20/final.w'
        if mul == 3:
            layer = nn.Sequential(
                SRNet3(mul),
                create_cnn_model(base_arch=self.base_arch, nc=self.n_classes)
            )
            layer[0].load_state_dict(torch.load(p_x3), strict=True) if Path(p_x3).is_file() else None
            layer[-1].load_state_dict(torch.load(p_vgg), strict=True) if Path(p_vgg).is_file() else None
            return layer
        elif mul == 2:
            layer = nn.Sequential(
                SRNet3(mul),
                create_cnn_model(base_arch=self.base_arch, nc=self.n_classes)
            )
            layer[0].load_state_dict(torch.load(p_x2), strict=True) if Path(p_x2).is_file() else None
            layer[-1].load_state_dict(torch.load(p_vgg), strict=True) if Path(p_vgg).is_file() else None
            return layer
        else:
            layer = nn.Sequential(
                PassThrough(),
                create_cnn_model(base_arch=self.base_arch, nc=self.n_classes)
            )
            layer[-1].load_state_dict(torch.load(p_vgg), strict=True) if Path(p_vgg).is_file() else None
            return layer

    def layer_groups(self):
        return [
            self.stn,
            self.pyramid_group
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
            x_out = o1
            branches_out.append(x_out)
        x_stack = torch.stack(branches_out, dim=-1)  # [bs, n_classes, n_branches]
        # weights = self.branch_weighted(x_stack.view([-1, self.n_classes * self.n_branches]))

        if not self.training and self.is_testing:
            # not when valid, just when test
            return weights_branches((x_stack, None))

        return x_stack, None


class PyramidSRShare(nn.Module):
    """
    Like `prlab.model.pyramid_sr.PyramidSRNonShare` but all branches share the basic layers of VGG, just different
    of first layer for different size of input and do not broken the weight.
    It is for special VGG
    """

    def __init__(self, **config):
        super().__init__()
        self.multiples = config.get('multiples', [1, 2])
        self.config = config
        self.n_classes = config['n_classes']

        # stn layer
        self.stn = STN(img_size=config['img_size'])

        # latest layer, classifier, main and shared layer
        self.base_arch = base_arch_str_to_obj(config.get('base_arch', 'vgg16_bn'))

        input_spec, body, head = self.separated_base_arch()
        # this part is share, not need to clone, but input_spec should be clone if used
        self.classifier = nn.Sequential(body, head)

        # some parallel branches, in different size with SR
        pyramid_group = []
        for mul in self.multiples:
            pyramid_group.append(self.make_layer_pyramid(mul, input_spec))

        self.pyramid_group = nn.Sequential(*pyramid_group)

        self.is_testing = False

    def separated_base_arch(self):
        """
        load weights if have and separated the base arch to three part, [first some layers, remain of base, head (fc)]
        note: if share then can reuse, but if some parts is not share then must make new and clone weights (first layers)
        :return:
        """
        base_weights_path = self.config.get('base_weights_path', None)
        base_model = create_cnn_model(base_arch=self.base_arch, nc=self.n_classes)
        o = base_model.load_state_dict(torch.load(base_weights_path), strict=False) \
            if base_weights_path is not None and Path(base_weights_path).is_file() else None
        if o:
            print('load weights from ', base_weights_path)

        if self.config['base_arch'] in ['vgg16_bn']:
            sep_pos = self.config.get('sep_pos', 3)  # 3 for first Conv2D layer, 6 for top 2 Conv2D layers
            out = [
                nn.Sequential(*base_model[0][0][:sep_pos]),
                nn.Sequential(*base_model[0][0][sep_pos:], base_model[0][1:]),
                nn.Sequential(*base_model[1:])
            ]
            return out
        elif self.config['base_arch'] in ['resnet101']:
            out = [
                nn.Sequential(*base_model[0][:3]),
                nn.Sequential(*base_model[0][3:]),
                nn.Sequential(*base_model[1:])
            ]
            return out
        else:
            raise Exception('Does not support yet for {}'.format(self.config['base_arch']))

    def make_layer_pyramid(self, mul, input_spec):
        """
        :param mul:
        :param input_spec: first layers, must adapt to input size
        :return:
        """
        srnet_c = SRNet3
        srnet_c_str = self.config.get('srnet_fn', self.config.get('srnet_c', None))
        if srnet_c_str is not None:
            srnet_c, _ = load_func_by_name(srnet_c_str)

        if mul in [2, 3]:
            # make new one here, similar input_spec but triple size input, copy weights
            layer = nn.Sequential(
                srnet_c(scale=mul),
                make_basic_block(input_spec.state_dict(), strict=True, module_like=input_spec)
            )
            path_xn = self.config.get('weight_path_x{}'.format(mul), None)
            o = layer[0].load_state_dict(torch.load(path_xn), strict=True) \
                if path_xn is not None and Path(path_xn).is_file() else None
            if o:
                print('load weights from ', path_xn)

        else:
            # just clone input_spec here
            layer = make_basic_block(input_spec.state_dict(), strict=True, module_like=input_spec)
        return layer

    def layer_groups(self):
        return [
            nn.Sequential(self.stn),
            self.classifier,
            self.pyramid_group
        ]

    def load_weights(self, **config):
        """ TODO not need yet, load when make model """

    def forward(self, *x, **kwargs):
        x = self.stn(x[0])

        branches_out = []
        for idx in range(len(self.pyramid_group)):
            o1 = self.pyramid_group[idx](x)
            x_out = self.classifier(o1)
            branches_out.append(x_out)
        x_stack = torch.stack(branches_out, dim=-1)  # [bs, n_classes, n_branches]

        if not self.training and self.is_testing:
            # not when valid, just when test
            return weights_branches((x_stack, None))

        return x_stack, None


class PyramidSRVGGShare(PyramidSRShare):
    """
    Wrap for PyramidSRShare with vgg architecture
    """

    def __init__(self, **config):
        config['base_arch'] = 'vgg16_bn'
        super().__init__(**config)


class PyramidShare(PyramidSRShare):
    """
    Provided the big size image (after resize) then use multi size to train.
    [low-size, SR-size+, big-size]
    many branches share some latest layers (sep_pos).
    Now support to VGG but can be extended in future.
    multiples should be [1, 2, 1], [1, 2, 4, 1]
    [1, 2, 4, 1] mean [low-size, SR 2 low-size, SR 4 low-size, big-size] and low-size = big-size/2**2
    sep_pos should be 10, 6 or 3
    """

    def __init__(self, **config):
        self.group_1, self.group_2 = [], []
        super().__init__(**config)

    def make_layer_pyramid(self, mul, input_spec):
        """
        :param mul:
        :param input_spec: first layers, must adapt to input size
        :return:
        """
        srnet_c = SRNet3
        srnet_c_str = self.config.get('srnet_fn', self.config.get('srnet_c', None))
        if srnet_c_str is not None:
            srnet_c, _ = load_func_by_name(srnet_c_str)

        if mul in [2, 3, 4]:
            # make new one here, similar input_spec but triple size input, copy weights
            layer = nn.Sequential(
                srnet_c(scale=mul),
                make_basic_block(input_spec.state_dict(), strict=True, module_like=input_spec)
            )
            self.group_1.append(layer[0]), self.group_2.append(layer[1:])
            path_xn = self.config.get('weight_path_x{}'.format(mul), None)
            o = layer[0].load_state_dict(torch.load(path_xn), strict=False) \
                if path_xn is not None and Path(path_xn).is_file() else None
            if o:
                print('load weights from ', path_xn, 'for x', mul)

        else:
            # just clone input_spec here
            layer = make_basic_block(input_spec.state_dict(), strict=True, module_like=input_spec)
            self.group_2.append(layer)
        return layer

    def layer_groups(self):
        return [
            nn.Sequential(self.stn, *self.group_1),
            nn.Sequential(self.classifier),
            nn.Sequential(*self.group_2),
            # self.pyramid_group
        ]

    def forward(self, *x, **kwargs):

        b, c, h, w = x[0].size()
        m_size = len(self.multiples)
        hs = (2 ** (m_size - 2))  # now /4, /2, ..., if /1.4 then need change it
        nh, nw = h // hs, w // hs
        x = self.stn(x[0])

        # x_small_size = resize_tensor(x, nh, nw)
        with torch.no_grad():
            x_small_size = (F.adaptive_avg_pool2d(x, (nh, nw))).data

        input_branches = [x_small_size] + [x_small_size] * (m_size - 2) + [x]

        branches_out = []
        for idx in range(len(self.pyramid_group)):
            o1 = self.pyramid_group[idx](input_branches[idx])
            x_out = self.classifier(o1)
            branches_out.append(x_out)
        x_stack = torch.stack(branches_out, dim=-1)  # [bs, n_classes, n_branches]

        if not self.training and self.is_testing:
            # not when valid, just when test
            return weights_branches((x_stack, None))

        return x_stack, None


class PyramidSRVGGShareDeeper(PyramidSRVGGShare):
    """
    Extend of `PyramidSRVGGShare` just make deeper (add some layer) when mul is 2 and 3.
    Mostly keep sep_pos=3 instead 6
    """

    def __init__(self, **config):
        super().__init__(**config)

    def make_layer_pyramid(self, mul, input_spec):
        """
        :param mul:
        :param input_spec: first layers, must adapt to input size
        :return:
        """
        if mul in [2, 3]:
            # make new one here, similar input_spec but triple size input, copy weights
            # in this case, input_spec mostly has len(3) and out filter is 64 (both VGG16 and ResNet101)
            layer = nn.Sequential(
                SRNet3(mul),
                make_basic_block(input_spec.state_dict(), strict=True, module_like=input_spec),
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                )
            )
            path_xn = self.config.get('weight_path_x{}'.format(mul), None)
            o = layer[0].load_state_dict(torch.load(path_xn), strict=True) \
                if path_xn is not None and Path(path_xn).is_file() else None
            if o:
                print('load weights from ', path_xn)

        else:
            # just clone input_spec here
            layer = make_basic_block(input_spec.state_dict(), strict=True, module_like=input_spec)
        return layer


def make_basic_block(state_dict=None, strict=True, module_like=None):
    """
    This block is widely used as the first block, in VGG16, ResNet101, e.g. (check)
    :param state_dict:
    :param strict:
    :param module_like: if given then new block will like the module (TODO try to deepcopy)
    :return:
    """
    block = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU()
    )
    if module_like and len(module_like) >= 6:
        # add more three submodule, the second layer
        block = nn.Sequential(
            *block,
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
    if module_like and len(module_like) >= 10:
        # add more three submodule, extends from 6 with 4 small part
        # just work for VGG
        block = nn.Sequential(
            *block,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
    if state_dict is not None:
        block.load_state_dict(state_dict=state_dict, strict=strict)

    return block


def weights_branches(pred, **kwargs):
    """
    Use together with `prlab.model.pyramid_sr.prob_weights_loss`
    :param pred: out, weights: [bs, n_classes, n_branches] [bs, n_branches]
    :param kwargs:
    :return:
    """
    if not isinstance(pred, tuple):
        # if not tuple then return the final/weighted version => DO NOTHING
        return pred

    out, _ = pred

    n_branches = out.size()[-1]
    sm = [torch.softmax(out[:, :, i], dim=1) for i in range(n_branches)]
    sm = torch.stack(sm, dim=-1)
    # c_out = torch.bmm(sm, weights.unsqueeze(-1)).squeeze(dim=-1)
    c_out = torch.mean(sm, dim=-1)

    return c_out


def prob_weights_loss(pred, target, **kwargs):
    """
    `CrossEntropyLoss` but for multi branches (out, weight) :([bs, C, branches], [bs, branches])
    :param pred:
    :param target:
    :param kwargs:
    :return:
    """
    f_loss, _ = load_func_by_name('prlab.fastai.utils.prob_loss_raw')
    if not isinstance(pred, tuple):
        return f_loss(pred, target)

    out, _ = pred
    n_branches = out.size()[-1]
    losses = [f_loss(out[:, :, i], target) for i in range(n_branches)]

    losses = torch.stack(losses, dim=-1)
    # loss = (losses * weights).sum(dim=-1)
    loss = torch.mean(losses, dim=-1)
    return torch.mean(loss)


def prob_weights_acc(pred, target, **kwargs):
    """
    Use together with `prlab.model.pyramid_sr.prob_weights_loss`
    :param pred: out, weights: [bs, n_classes, n_branches] [bs, n_branches]
    :param target: int for one-hot and list of float for prob
    :param kwargs:
    :return:
    """
    f_acc, _ = load_func_by_name('prlab.fastai.utils.prob_acc')
    c_out = weights_branches(pred=pred)

    return f_acc(c_out, target)  # f_acc(pred[0][:, :, 0], target)


def norm_weights_loss(pred, target, **kwargs):
    """
    `CrossEntropyLoss` but for multi branches (out, weight) :([bs, C, branches], [bs, branches])
    :param pred:
    :param target:
    :param kwargs:
    :return:
    """
    # f_loss, _ = load_func_by_name('prlab.fastai.utils.prob_loss_raw')
    f_loss = nn.CrossEntropyLoss()
    if not isinstance(pred, tuple):
        return f_loss(pred, target)

    out, _ = pred
    n_branches = out.size()[-1]
    losses = [f_loss(out[:, :, i], target) for i in range(n_branches)]

    losses = torch.stack(losses, dim=-1)
    # loss = (losses * weights).sum(dim=-1)
    loss = torch.mean(losses, dim=-1)
    return torch.mean(loss)


def norm_weights_acc(pred, target, **kwargs):
    """
    Use together with `prlab.model.pyramid_sr.prob_weights_loss`
    :param pred: out, weights: [bs, n_classes, n_branches] [bs, n_branches]
    :param target: int for one-hot and list of float for prob
    :param kwargs:
    :return:
    """
    f_acc, _ = load_func_by_name('fastai.metrics.accuracy')
    c_out = weights_branches(pred=pred)

    return f_acc(c_out, target)  # f_acc(pred[0][:, :, 0], target)
