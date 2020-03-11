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
    config['label_names'] = np.array(ferlabels)
    return config
