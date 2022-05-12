#!/usr/bin/env python3

import numpy as np
import plottools
import matplotlib.pyplot as plt

plottools.apply_params(plt)
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'lines.linewidth': 1})
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
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

def find(name):
    f = glob(os.path.join(args.datadir, name))
    if len(f):
        return f[0]
    return None

files = [
    (find("cavity_N*_Re100_adam.csv"), '--', 'C0'),
    (find("cavity_N*_Re100_lbfgsb.csv"), '--', 'C1'),
    (find("cavity_N*_Re100_newton.csv"), '--', 'C2'),
    (find("cavity_N*_Re400_adam.csv"), '-', 'C0'),
    (find("cavity_N*_Re400_lbfgsb.csv"), '-', 'C1'),
    (find("cavity_N*_Re400_newton.csv"), '-', 'C2'),
]

files = [f for f in files if f[0] is not None]

print("Loading data from files:")
for f in files:
    print(f[0])

figsize = (1.8, 1.6)
fig, ax = plt.subplots(figsize=figsize)
ax.set_xscale('log')
ax.set_yscale('log')
for f, ls, lc in files:
    u = np.genfromtxt(f, names=True, delimiter=',')
    label = os.path.basename(f)
    ax.plot(u['epoch'], u['ref_du_l2'], ls=ls, c=lc, label=label)
ax.set_xlabel('epoch')
ax.set_xlim(1, 1e5)
ax.set_xticks([1, 1e1, 1e2, 1e3, 1e4, 1e5])
ax.set_ylabel('error')
ax.set_ylim(1e-6, 1)
ax.yaxis.set_label_coords(-.24, .5)
plottools.savefig(fig, 'error.pdf', bbox_inches='tight', pad_inches=0.01)
plottools.savelegend(fig, ax, 'legend.pdf')
plt.close(fig)
