#!/usr/bin/env python3

import numpy as np
import re
import os
import json
from glob import glob
from util_cache import cache_to_file
from argparse import Namespace
import pickle

import plottools
import matplotlib.pyplot as plt
plottools.apply_params(plt)
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'lines.linewidth': 1})

descs = [
    ('fd_newton', 'fd_newton', 'C2', [
        'errorline',
    ]),
    ('fd_lbfgsb', 'fd_lbfgsb', 'C1', []),
    ('pinn_lbfgsb', 'pinn_lbfgsb', 'C0', ['errorline', 'scatter']),
]


def get_median(lines, xx):
    xxunique = np.unique(xx)
    indices = dict()
    for x in xxunique:
        indices[x] = np.where(xx == x)
    vlines = vars(lines)
    res = Namespace()
    for key in vlines:
        u = np.empty(len(xxunique))
        for i, x in enumerate(xxunique):
            ux = vlines[key][indices[x]]
            if len(ux):
                u[i] = np.median(ux)
            else:
                u[i] = np.nan
        res.__setattr__(key, u)
    return res


def plot(descs, lines_all, figsize=(1.7, 1.5)):
    mlines_all = []
    for lines in lines_all:
        mlines_all.append(get_median(lines, lines.N))

    xlim = (100, 1e5)
    xticks = [1e2, 1e3, 1e4, 1e5]

    flags = [desc[3] for desc in descs]
    labels = [desc[1] for desc in descs]
    colors = [desc[2] for desc in descs]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    for i, lines in enumerate(lines_all):
        mlines = mlines_all[i]
        if 'errorline' in flags[i]:
            ax.plot(mlines.N, mlines.error, c=colors[i], label=labels[i])
        if 'scatter' in flags[i]:
            ax.scatter(lines.N,
                       lines.error,
                       marker='.',
                       s=5,
                       alpha=0.5,
                       facecolor='k',
                       edgecolor='none')
    lN = np.linspace(*xlim, 100)
    ax.plot(lN, 39 * lN**(-1.), c='k', ls='--', lw=1, zorder=-5)
    ax.set_xlabel('parameters N')
    ax.set_ylabel('error')
    ax.set_xlim(xlim)
    ax.set_xticks([1e2, 1e3, 1e4, 1e5])
    ax.set_ylim(1e-4, 1)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    plottools.savefig(fig, 'error.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    for i, lines in enumerate(lines_all):
        mlines = mlines_all[i]
        ax.plot(mlines.N, mlines.signif_time, c=colors[i], label=labels[i])
        if 'scatter' in flags[i]:
            ax.scatter(lines.N,
                       lines.signif_time,
                       marker='.',
                       s=5,
                       alpha=0.5,
                       facecolor='k',
                       edgecolor='none',
                       label=labels[i])
    ax.set_xlabel('parameters N')
    ax.set_ylabel('execution time [s]')
    ax.set_xlim(xlim)
    ax.set_xticks([1e2, 1e3, 1e4, 1e5])
    ax.set_ylim(1e-3, 1e6)
    plottools.savefig(fig, 'signif_time.pdf')
    plottools.savelegend(fig, ax, 'legend.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    for i, lines in enumerate(lines_all):
        mlines = mlines_all[i]
        ax.plot(mlines.N, mlines.signif_epochs, c=colors[i], label=labels[i])
        if 'scatter' in flags[i]:
            ax.scatter(lines.N,
                       lines.signif_epochs,
                       marker='.',
                       s=5,
                       alpha=0.5,
                       edgecolor='none',
                       facecolor='k',
                       label=labels[i])
    N = np.linspace(lines.N.min(), lines.N.max(), 100)
    ax.set_xlabel('parameters N')
    ax.set_ylabel('epochs')
    ax.set_xlim(xlim)
    ax.set_xticks([1e2, 1e3, 1e4, 1e5])
    ax.set_yticks([1e2, 1e3, 1e4, 1e5])
    ax.set_ylim(100, 1e5)
    plottools.savefig(fig, 'signif_epochs.pdf')
    plt.close(fig)


def main():
    lines_all = []
    for i in range(len(descs)):
        path = descs[i][0]
        with open(os.path.join(path, "lines.pickle"), 'rb') as f:
            lines = pickle.load(f)
            lines_all.append(lines)
    plot(descs, lines_all)


if __name__ == "__main__":
    main()
