# fmap_name = lambda o: o # for normal train folder
'''ang dis exc fea fru hap neu oth sad sur xxx'''
import os
from pathlib import Path

import pandas as pd
# default_emo_lst = ['ang', 'hap', 'neu', 'sad', 'exc']
# four_class_emo_lst = ['ang', 'hap', 'neu', 'sad']
# lbl2id = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
from fastai.data_block import FloatList

from prlab.fastai.data_funcs import DefaultDataHelper
from prlab.gutils import map_str_prob


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
