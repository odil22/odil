#!/usr/bin/env python3

from argparse import Namespace
import numpy as np
import pickle
import os
import sys

import plottools
import matplotlib.pyplot as plt
plottools.apply_params(plt)
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'lines.linewidth': 1})


def plot(thres = 0.05):
    path_data = 'data.pickle'
    with open(path_data, 'rb') as f:
        data = pickle.load(f)

    seed = np.array([d.seed for d in data])
    nimp = np.array([d.nimp for d in data])
    error_linf = np.array([d.error_linf for d in data])
    error_l1 = np.array([d.error_l1 for d in data])
    error_l2 = np.array([d.error_l2 for d in data])


    fig, ax = plt.subplots(figsize=(1.7, 1.5))

    xx = nimp
    xxm = np.unique(xx)
    yy = error_l1
    yym = np.array([np.nanmin(yy[np.where(xx == x)]) for x in xxm])
    yymed = np.array([np.nanmedian(yy[np.where(xx == x)]) for x in xxm])


    ax.scatter(xx, yy, s=2, edgecolor='none', facecolor='k', alpha=0.1)
    ax.plot(xxm, yym, c='C0')
    #ax.plot(xxm, yymed, c='k')

    sel = np.where(yym < thres)[0]
    imin = -1
    if len(sel):
        imin = np.min(sel)
        ax.axvline(xxm[imin], c='k', ls='--')

    ax.set_yscale('log')
    ax.set_xlabel('points K')
    ax.set_ylabel('error')
    ax.set_xlim(0, 50)
    ax.set_xticks([0, 25, 50])
    ax.set_ylim(None, 1)

    if yym.min() < 1e-10:
        ax.set_ylim(1e-16, 1)
        ax.set_yticks([10**(-p) for p in [0, 4, 8, 12, 16]])

    name = os.path.basename(os.path.abspath(os.getcwd()))
    if name.startswith("vary_"):
        name = name[5:]
    rename = {'pois' : 'poiseuille'}
    name = rename.get(name, name)
    ax.set_title(r'{} , $K_\mathrm{{min}}({:})={:}$'.format(
        name, thres, xxm[imin] if imin >= 0 else 0))

    plottools.savefig(fig, "error.pdf")

if __name__ == "__main__":
    plot()
