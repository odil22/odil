#!/usr/bin/env python3

import json
import csv
from glob import glob
import os
from argparse import Namespace
import numpy as np
import pickle


def load_data(path):
    dirs = sorted(glob(os.path.join(path, "*/args.json")))
    dirs = list(map(os.path.dirname, dirs))
    res = []
    for i, d in enumerate(dirs):
        if i % 100 == 0:
            print('{:}/{:}: {}'.format(i + 1, len(dirs), d))
        item = Namespace()
        path_args = os.path.join(d, "args.json")
        path_train = os.path.join(d, "train.csv")
        path_imposed = os.path.join(d, "imposed.csv")
        if not os.path.isfile(path_args) or not os.path.isfile(path_train):
            print("skip unfinished '{}'".format(d))
            continue
        try:
            with open(path_args, 'r') as f:
                args = Namespace(**json.load(f))
        except:
            continue

        try:
            csv = np.genfromtxt(path_train,
                                names=True,
                                delimiter=',')
            csv = csv[-1]
        except:
            continue

        try:
            imposed = np.genfromtxt(path_imposed,
                                    names=True,
                                    delimiter=',')
        except:
            continue

        item.nimp = args.nimp
        item.seed = args.seed
        item.error_linf = max(csv['ref_du_linf'], csv['ref_dv_linf'])
        item.error_l1 = abs(csv['ref_du_l1']) + abs(csv['ref_dv_l1'])
        item.error_l2 = (csv['ref_du_l2'] ** 2 + csv['ref_dv_l2'] ** 2) ** 0.5
        item.imposed = imposed
        res.append(item)
    return res

if __name__ == '__main__':
    path = 'data.pickle'
    with open(path, 'wb') as f:
        data = load_data('.')
        print(path)
        pickle.dump(data, f, protocol=4)
