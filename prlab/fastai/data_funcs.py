# Implement many help functions for data loading in DataBunch of fastai

import os
from pathlib import Path


# --------------------------------------------------------------------------
# For RAF-DB data
# --------------------------------------------------------------------------

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
    return raf_meta.loc[name, 'is_test'] is False


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
