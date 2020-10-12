# Implement many help functions for data loading in DataBunch of fastai

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
# --------------------------------------------------------------------------
# For RAF-DB data
# --------------------------------------------------------------------------
from fastai.data_block import CategoryList, FloatList

from prlab.emotion.emo_const import rafdb_labels_names, rafdb_emo_dis, affectnet_emo_dis
from prlab.common.utils import set_if, balanced_sampler


class DefaultDataHelper:
    label_cls = CategoryList

    def __init__(self, **kwargs):
        pass

    def y_func(self, o):
        return 0

    def filter_func(self, file_path):
        return True


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


class RafDBDataHelperDis(RafDBDataHelper):
    label_cls = FloatList

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rafdb_name_2_id = {name: idx for idx, name in enumerate(rafdb_labels_names)}

    def y_func(self, o):
        lbl_name = super().y_func(o)
        lbl_id = self.rafdb_name_2_id[lbl_name]
        return rafdb_emo_dis[lbl_id]


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


# -------------------------------------------------------------------------
# Functions for AffectNet dataset
# -------------------------------------------------------------------------
def affect_net_name_norm_fn(x): return '_'.join(x.split('_')[1:])


def affect_net_name_norm_x_fn(x): return x.replace('/', '_')


class AffectNetDataHelper(DefaultDataHelper):
    """
    Data Helper for AffectNet, categorical output 7/8 emotion classes
    class order: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt (latest 3 removed)
    Contempt should be use or not (to compare) then 7 or 8 classes
    """
    label_cls = CategoryList

    def __init__(self, **config):
        super().__init__(**config)
        self.n_classes = config.get('n_classes', 8)
        self.classes = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'][:self.n_classes]

    def y_func(self, file_path):
        label = (file_path.parts[-2] if isinstance(file_path, Path) else file_path.split(os.path.sep)[-2])
        return label

    def filter_func(self, file_path):
        label = self.y_func(file_path=file_path)
        if isinstance(label, str):
            label = int(label)
        return label < self.n_classes


class AffectNetDataHelperDis(AffectNetDataHelper):
    label_cls = FloatList

    def __init__(self, **config):
        super().__init__(**config)

    def y_func(self, file_path):
        lbl = super().y_func(file_path=file_path)
        int_lbl = int(lbl)
        int_lbl = int_lbl if int_lbl < self.n_classes else 0
        # convert to dis
        return affectnet_emo_dis[int_lbl][:self.n_classes]

    def filter_func(self, file_path):
        label = self.y_func(file_path=file_path)
        label = np.argmax(label)
        return label < self.n_classes


class AffectNetBalanceValDataHelper(AffectNetDataHelper):
    """
    Extend `AffectNetDataHelper` with a valid function to split.
    Valid is random select at the beginning and keep
    """
    _map_name_fn = affect_net_name_norm_x_fn

    def __init__(self, **config):
        super().__init__(**config)

        self.valid_names = set([])
        self._build_valid_name(**config)

    def y_func(self, file_path):
        # override super class function, because it should be call from df, that not actually load from folder name
        # then can extract from filename itself
        # format of affectnet: label_name
        file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        name = file_path.name
        label = name.split('_')[0]
        return label

    def valid_func(self, fname):
        fname = fname if isinstance(fname, Path) else Path(fname)
        name = fname.name
        return name in self.valid_names

    def _build_valid_name(self, **config):
        csv_names = config['meta_csv']
        csv_names = csv_names if isinstance(csv_names, list) else [csv_names]

        # merge 2 df
        dfs = [pd.read_csv(f_name) for f_name in csv_names]
        df_merge = pd.concat(dfs, axis=0, ignore_index=True)

        map_names = ['{}_{}'.format(lbl, AffectNetBalanceValDataHelper._map_name_fn(o)) for lbl, o in
                     zip(df_merge['expression'], df_merge['subDirectory_filePath'])]
        labels = [self.y_func(o) for o in map_names]
        selected_pos = balanced_sampler(labels=labels, n_each=config.get('n_validation_each_class', 1000),
                                        replacement=False)
        self.valid_names = [map_names[p] for p in selected_pos]
        self.valid_names = set(self.valid_names)  # set for quick check "IN" operator


class AffectNetDataHelperReg(DefaultDataHelper):
    """
    Data Helper for AffectNet, arousal and/or valency.
    When run with it, configure should set n_classes=1/2 for regression
    """
    label_cls = FloatList

    # mode for label
    _BOTH_MODE = 0
    _ONLY_VALENCY = 1
    _ONLY_AROUSAL = 2

    def __init__(self, **config):
        """
        meta_csv is csv or list of csv files, must same columns order and length (mostly training.csv and validation.csv)
        :param config:
        """
        super().__init__(**config)
        self.label_mode = config.get('label_mode', self._BOTH_MODE)

        self.name_norm_fn = lambda x: '_'.join(x.split('_')[1:])
        self.name_norm_x_fn = lambda x: x.replace('/', '_')

        self.df = self.read_csvs(**config)

    def y_func(self, file_path):
        file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        name_norm = self.name_norm_fn(file_path.name)
        valence, arousal = self.df.loc[name_norm, ['valence', 'arousal']]

        if self.label_mode == self._BOTH_MODE:
            return [valence, arousal]
        elif self.label_mode == self._ONLY_VALENCY:
            return valence
        elif self.label_mode == self._ONLY_AROUSAL:
            return arousal

        return 0  # error

    def filter_func(self, file_path):
        """
        Remove -2 values, why, some row has -2 in both valency and arousal
        :param file_path:
        :return:
        """
        try:
            x = self.y_func(file_path)
            x = x[0] if isinstance(x, list) else x
            return -1. <= x <= 1.
        except:
            # files where outside of training and validation folder should not found in df
            return False

    def read_csvs(self, **config):
        idx_name = config.get('new_index_name', 'name')
        csv_names = config['meta_csv']
        csv_names = csv_names if isinstance(csv_names, list) else [csv_names]

        # merge 2 df
        dfs = [pd.read_csv(f_name) for f_name in csv_names]
        df_merge = pd.concat(dfs, axis=0, ignore_index=True)

        # make index after change subDirectory_filePath to name_norm
        idx_col = [self.name_norm_x_fn(o) for o in df_merge['subDirectory_filePath']]
        idx_col = pd.DataFrame(idx_col, columns=[idx_name])
        df_merge.set_index('subDirectory_filePath')

        df = pd.concat([df_merge, idx_col], axis=1)
        df.set_index(idx_name, drop=True, inplace=True)

        return df

# -------------------------------------------------------------------------
# Functions for AffectNet dataset
# -------------------------------------------------------------------------
