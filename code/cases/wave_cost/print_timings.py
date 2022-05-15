#!/usr/bin/env python3

import numpy as np
import os
import json
from glob import glob
from plot_lines_all import get_median
import pickle

for path in sorted(glob("*/lines.pickle")):
    with open(path, 'rb') as f:
        lines = pickle.load(f)
        mlines = get_median(lines, lines.N)
        print(path)
        N0 = 20000
        i = np.argmin(abs(lines.N - N0))
        print("N={:} t={:.2f} s".format(mlines.N[i], mlines.signif_time[i]))
