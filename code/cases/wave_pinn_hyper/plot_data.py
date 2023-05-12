#!/usr/bin/env python3

import numpy as np
import re
import os
import json
from glob import glob
from argparse import Namespace
import argparse
import pickle


def read_data(filesdone):
    res = dict()
    for i, fd in enumerate(filesdone):
        basedir = os.path.dirname(fd)
        if i % 10 == 0:
            print("{:}/{:} {}".format(i + 1, len(filesdone), basedir))
        train = np.genfromtxt(os.path.join(basedir, 'train.csv'),
                              names=True,
                              delimiter=',')
        train = np.atleast_1d(train)
        res[basedir] = dict()
        res[basedir]['epoch'] = train['epoch'][-1]
        res[basedir]['loss'] = train['loss'][-1]
        res[basedir]['u_err_l2'] = train['u_err_l2'][-1]
    return res


def plot(lines, prefix=''):
    import matplotlib.pyplot as plt
    import plotutil
    import scipy
    import scipy.stats
    plotutil.set_extlist(['pdf'])
    figsize = (2.0, 1.3)
    fig, ax = plt.subplots(figsize=figsize)
    verror = []
    vlabel = []
    vcolor = []
    for line in lines:
        _, _, name, color, data = line
        error = [entry['u_err_l2'] for entry in data.values()]
        logerror = np.log10(error)
        verror.append(error)
        vlabel.append(name)
        vcolor.append(color)
        xx = np.linspace(*args.xlim, 300)
        density = np.mean(
            [scipy.stats.norm.pdf(xx, loc=x, scale=0.01) for x in logerror],
            axis=0)
        #bins = np.logspace(-1.5, -0.5, 21)
        #ax.hist(verror, bins=bins, histtype='step', color=vcolor, label=vlabel, fill=False, lw=0.75)
        ax.plot(xx, density, c=color, label=name)
    ax.set_xlabel(r'$\log_{10} \mathrm{error}$')
    ax.set_ylabel('probaility density')
    #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    prefix = '_' + prefix.strip('_') if prefix else ''
    plotutil.savefig(fig, os.path.join('error_hist' + prefix))
    plotutil.savelegend(fig, ax, 'error_hist_leg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', type=int, default=1)
    parser.add_argument('--xlim', type=float, nargs=2, default=[-1.5, -0.5])
    parser.add_argument('--extract', type=int, default=0)
    parser.add_argument('--prefix', type=str, default='')
    args = parser.parse_args()
    prefix = args.prefix
    # Suffix, name, color.
    lines = [
        [prefix + 'baseline', 'baseline', 'C0'],
        [prefix + 'norm', 'normalized', 'C1'],
        [prefix + 'norm_glorot', 'normalized, glorot', 'C2'],
    ]
    # Convert to: outdir, pickle, name, color.
    lines = [['out_{}'.format(line[0]), 'data_{}.pickle'.format(line[0])] +
             line[1:] for line in lines]
    if args.extract:
        for line in lines:
            outdir = line[0]
            filesdone = sorted(glob(outdir + "/out_*/done"))
            data = read_data(filesdone)
            if not len(data):
                print("skip empty '{}'".format(line[0]))
                continue
            path = line[1]
            with open(path, 'wb') as f:
                print(path)
                pickle.dump(data, f)
    if args.plot:
        for line in lines:
            path = line[1]
            with open(path, 'rb') as f:
                print(path)
                data = pickle.load(f)
            line.append(data)
        plot(lines, prefix=prefix)
