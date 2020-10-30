"""
Implement some pipeline func support `Pipeline Process template` in `prlab.fastai.pipeline.pipeline_control`
"""
from fastai.vision import *

from prlab.emotion.emo_const import ferlabels


def pre_processing(**config):
    """
    implement some pre_processing for config before load data and train.
    support `Pipeline Process template`
    should be add before any data load step in the pipeline.
    :param config:
    :return:
    """
    config['label_names'] = np.array(ferlabels[:config['n_classes']])
    return config


def rotate_img_size(**config):
    """
    `Pipeline Process template`
    TO ratate img_size (input image size) in config in order
    :param config: img_size, img_size_rotate
    :return: config with update img_size
    """
    rotate_size = config['img_size_rotate']
    cur_pos = [i for i in range(len(rotate_size)) if rotate_size[i] == config['img_size']]
    if len(cur_pos) == 1:
        cur_pos = cur_pos[0]
        next_pos = cur_pos + 1 if cur_pos + 1 < len(rotate_size) else 0
        config['img_size'] = rotate_size[next_pos]
    else:
        "Do nothing, error here: 0 -> size is not in list, >1: multi same value"

    return config
