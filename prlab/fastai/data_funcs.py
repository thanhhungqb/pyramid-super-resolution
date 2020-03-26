# Implement many help functions for data loading in DataBunch of fastai

import os
import random
from pathlib import Path

import pandas as pd
# --------------------------------------------------------------------------
# For RAF-DB data
# --------------------------------------------------------------------------
from fastai.data_block import CategoryList

from prlab.gutils import set_if


class DefaultDataHelper:
    label_cls = CategoryList

    def __init__(self, **kwargs):
        pass

    def y_func(self, o):
        return 0


class RafDBDataHelper(DefaultDataHelper):
    """
    Data Helper for rafDB
    """
    label_cls = CategoryList

    def __init__(self, **config):
        super().__init__(**config)
        set_if(config, 'csv_path', config['path'] / 'raf-db-meta.csv')

        self.raf_meta = pd.read_csv(config['csv_path'], index_col=0)
        all_fold = list(set(self.raf_meta['5_fold']))
        v = all_fold[int(random.random() * len(all_fold))]
        set_if(config, 'fold', v)

        self.f_map = fmap_name_aligned
        self.path = config['path'] / 'aligned'
        if not config.get('is_aligned', True):
            # use raw, default will use aligned above
            self.f_map = fmap_name
            self.path = config['path'] / 'train'

        self.fold = config['fold']

    def y_func(self, o):
        return raf_db_get_target_func(self.raf_meta, o, self.f_map)

    def filter_train_fn(self, o):
        """
        :param o:
        :return: true if train (and valid)/false for test
        """
        return raf_db_filter_train_func(self.raf_meta, o, map_fname_funcs=self.f_map)

    def filter_test_fn(self, o):
        """
        :param o:
        :return: true if test/false if train/valid
        """
        return not self.filter_train_fn(o)

    def split_valid_fn(self, o):
        """
        :param o:
        :return: true if in valid/false if not (then in train)
        """
        return raf_db_valid_split_func(self.raf_meta, o, fold=self.fold, map_fname_funcs=self.f_map)


fmap_name = lambda o: o  # for normal train folder
fmap_name_aligned = lambda o: '_'.join(o.split('_')[:2]) + ".jpg"  # for name in aligned


def raf_db_get_target_func(raf_meta, file_path, map_fname_funcs=lambda o: o):
    """
    can be wrapped outside to use with `label_from_func` with only one param $fname

    get_target_func = lambda fname: ferlbl2id[raf_db_get_target_func(df, fname)]

    or

    get_target_func = lambda fname: ferlbl2id[raf_db_get_target_func(df, fname, func)]

    :param raf_meta: df include 'emotion_name'
    :param file_path:
    :param map_fname_funcs: o->o for train and o->o cut aligned
    :return: label
    """
    name = (file_path.name if isinstance(file_path, Path) else file_path.split(os.path.sep)[-1])
    name = map_fname_funcs(name)
    return raf_meta.loc[name, 'emotion_name'].lower()


def raf_db_filter_train_func(raf_meta, file_path, map_fname_funcs=lambda o: o):
    """
    can be wrapped outside to use with `filter_by_func` with only one param $fname

    filter_func = lambda fname: raf_db_filter_train_func(df, fname)

    or

    filter_func = lambda fname: raf_db_filter_train_func(df, fname, o)

    :param raf_meta: df include 'is_test' that True/False
    :param file_path:
    :param map_fname_funcs: same as `raf_db_get_target_func`
    :return:
    """
    name = (file_path.name if isinstance(file_path, Path) else file_path.split(os.path.sep)[-1])
    name = map_fname_funcs(name)
    return not raf_meta.loc[name, 'is_test']


def raf_db_valid_split_func(raf_meta, file_path, fold=1, map_fname_funcs=lambda o: o):
    """
    Train/valid split by 5_fold column in df.
    Use to fix the train/valid set (not changed time by time) and may be use to k_fold

    Use with split_by_valid_func

    Must be wrap outside:
    valid_split = lambda o: raf_db_valid_split_func(df, o)
    
    :param raf_meta: df, include 5_fold column
    :param file_path:
    :param fold: fold id in 5_fold column use for valid
    :param map_fname_funcs: same as `raf_db_get_target_func`
    :return:
    """
    name = (file_path.name if isinstance(file_path, Path) else file_path.split(os.path.sep)[-1])
    name = map_fname_funcs(name)
    return raf_meta.loc[name, '5_fold'] == fold


# -------------------------------------------------------------------------
# END OF RAF-DB
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Functions for emotiw dataset
# -------------------------------------------------------------------------

def emotiw_get_target_func(fname):
    """
    Maybe wrap outside to use with `label_from_func`

    target_func = lambda o: emowlbl2id[emotiw_get_target_func(o)]

    Get target (label) for fname, this is parent of parent of this file
    label/movie.id/fname
    :param fname: Path or str to frame file
    :return: label by number, see emo_const to map to label
    """
    name = (fname.parts if isinstance(fname, Path) else fname.split(os.path.sep))[-3]
    return name.lower()

# -------------------------------------------------------------------------
# Functions for emotiw dataset
# -------------------------------------------------------------------------
