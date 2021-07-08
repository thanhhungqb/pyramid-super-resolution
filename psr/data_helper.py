# fmap_name = lambda o: o # for normal train folder
'''ang dis exc fea fru hap neu oth sad sur xxx'''
import os
from pathlib import Path

import numpy as np
import pandas as pd
# default_emo_lst = ['ang', 'hap', 'neu', 'sad', 'exc']
# four_class_emo_lst = ['ang', 'hap', 'neu', 'sad']
# lbl2id = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
from fastai.data_block import FloatList

from psr.data_funcs import DefaultDataHelper
from prlab.gutils import map_str_prob, get_file_rec


# def map_name_func(o): return o.stem if isinstance(o, Path) else Path(o).stem


# def map_exc_hap_func(o): return o if o != 'exc' else 'hap'


class FerplusDataHelper(DefaultDataHelper):
    label_cls = FloatList

    def __init__(self, csv_path, **kwargs):
        # self.df = df
        super().__init__(**kwargs)
        ferplus_meta = pd.read_csv(csv_path, delimiter=',')
        float_pro = map_str_prob(ferplus_meta['pro'])

        new_col = pd.DataFrame({'float_pro': [o for o in float_pro]})
        ferplus_meta = ferplus_meta.join(new_col)
        ferplus_meta = ferplus_meta.set_index(['Image name'])
        ferplus_meta = ferplus_meta.loc[~ferplus_meta.index.duplicated(keep='first')]

        self.ferplus_meta = ferplus_meta
        self.float_pro = float_pro

    def y_func(self, o):
        n = (o.name if isinstance(o, Path) else o.split(os.path.sep)[-1])
        return self.ferplus_meta.loc[n, :][-1]

    def get_y_func_ferplus(self, o):
        """alias for `y_func`"""
        return self.y_func(o)


def calc_train_emotion_distribution(csv_path, train_path=None, path=None, **kwargs):
    """
    calculate the emotion distribution on the training set
    :param csv_path: ferplus meta file
    :param train_path: path of train folder
    :param path: if train_path is None then get path should be given the base name
    :param kwargs:
    :return: {emotion_name: array}, {id: array}
    """
    if train_path is None:
        train_path = path / 'train' if isinstance(path, Path) else Path(path) / 'train'

    train_path = Path(train_path)
    train_names = [o.name for o in get_file_rec(train_path)]

    dh = FerplusDataHelper(csv_path=csv_path)
    df = dh.ferplus_meta
    print(df.head())

    count = {}
    su = {}
    s_0 = np.zeros(8)
    mapemo = {}  # map id and name
    for name in train_names:
        row = df.loc[name, :]
        e, en, lst = row['femotion'], row['nemotion'], row['float_pro']
        mapemo[e] = en
        count[e] = 1 + count.get(e, 0)
        su[e] = su.get(e, s_0) + np.array(lst)

    ret = {}
    for i in range(8):
        su[i] = su[i] / count[i]
        ret[mapemo[i]] = su[i]
        print(su[i])

    return ret, su
