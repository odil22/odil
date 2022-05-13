#!/usr/bin/env python3

import numpy as np
from collections import defaultdict
import re
import os
import json
from glob import glob
import argparse
from argparse import Namespace

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default='data')
args = parser.parse_args()
files = sorted(glob(os.path.join(args.datadir, "*.csv")))

for path in files:
    u = np.genfromtxt(path, names=True, delimiter=',')
    time = u['tt_opt']
    epoch = u['epoch']
    n = min(len(epoch), 1) + 1
    ne = epoch[-1] - epoch[-n]
    name = os.path.basename(path)
    print("{}, time per epoch {:.5f}s, epochs range {:}".format(
        name, (time[-1] - time[-n]) / (epoch[-1] - epoch[-n]), ne))
