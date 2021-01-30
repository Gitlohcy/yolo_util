
import numpy as np
import cv2
from pathlib import Path
import pandas as pd

from itertools import chain
from shutil import copyfile
from pprint import pprint
from tqdm.notebook import tqdm
from collections import defaultdict
from pdb import set_trace as st
from typing import *
from IPython.display import display
import pyperclip as ppc
import yaml


def ls(p, r='*'):
    return list(p.glob(r))

def lls(p, r='*'):
    return len(ls(p ,r))

def flat(to_chain: list) -> list:
    return list(chain(*to_chain))

def rename_dict(dict_, rename_keys: dict):
    for old_key, new_key in rename_keys.items():
        dict_[new_key] = dict_.pop(old_key)
    return dict_
    

def is_key_match(dict_1, dict_2):
    dict_1_keys = pd.Series(list(dict_1.keys()))
    dict_2_keys = pd.Series(list(dict_2.keys()))
    return dict_1_keys.isin(dict_2_keys).all()

def mkdir_r(dir_path):
    for p in [*reversed((dir_path.parents)), dir_path]:
        if not p.is_dir():
            p.mkdir()

def len_dict(dict_):
    return {k: len(list_) for k, list_ in dict_.items()}

def pd_len_dict(dict_list: List[dict]) -> pd.DataFrame:
    display(pd.DataFrame([len_dict(d) for d in dict_list]))
    
def arr_map(func, arr):
    return np.array(list(map(func, arr)))

def dump_yaml(dict_, fpath):
    with open(str(fpath), 'w') as f:
        yaml.dump(dict_, f, sort_keys=False)

def load_yaml(fpath):
    with open(str(fpath)) as f:
        dict_ = yaml.load(f, Loader=yaml.FullLoader)
        return dict_

#pd dataframe
def df_display_beside(dfs, names=[]):
    html_str = ''
    if names:
        html_str += ('<tr>' + 
                     ''.join(f'<td style="text-align:center">{name}</td>' for name in names) + 
                     '</tr>')
    html_str += ('<tr>' + 
                 ''.join(f'<td style="vertical-align:top"> {df.to_html(index=False)}</td>' 
                         for df in dfs) + 
                 '</tr>')
    html_str = f'<table>{html_str}</table>'
    html_str = html_str.replace('table','table style="display:inline"')
    display_html(html_str, raw=True)


Path.ls = ls
Path.lls = lls