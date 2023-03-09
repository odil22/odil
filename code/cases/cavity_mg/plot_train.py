#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import argparse
from argparse import Namespace
from glob import glob
import os
import re

import plotutil

def calc_loss(u):
    s = np.zeros_like(u['loss'])
    for i in range(10):
        k = 'loss' + str(i)
        if k in u.dtype.names:
            s += u[k]**2
    u['loss'] = s**0.5
    return u


def calc_epochtime(u, width=3):
    t = u['walltime']
    e = u['epoch']
    tm = np.hstack([[np.nan] * width, t[:-width]])
    em = np.hstack([[np.nan] * width, e[:-width]])
    res = np.empty(u.shape, dtype=u.dtype.descr + [('epochtime', float)])
    for name in u.dtype.names:
        res[name] = u[name]
    # Time of one epoch in milliseconds.
    res['epochtime'] = (t - tm) / (e - em) * 1000
    return res


def plot(lines, args):
    uu = [np.genfromtxt(line[0], delimiter=',', names=True) for line in lines]
    uu = [calc_loss(u) for u in uu]
    uu = [calc_epochtime(u, width=args.timewidth) for u in uu]
    for key, name, ylbl in [
        ('loss', 'loss', 'loss'),
        ('u_err_l2', 'err', 'error'),
        ('epochtime', 'epochtime', 'epoch time [ms]'),
    ]:
        if key not in uu[0].dtype.names:
            print("skip unknown key='{}'".format(key))
            continue
        fig, ax = plt.subplots()
        for i, line in enumerate(lines):
            lbl = '{:} levels'.format(line[1])
            u = uu[i]
            ax.plot(u['epoch'] + 1, u[key], label=lbl)
        ax.set_xlabel('epoch')
        ax.set_xscale('log')
        ax.set_xticks(10 ** np.arange(0, 5.1))
        plotutil.set_log_ticks(ax.xaxis)

        ax.set_ylabel(ylbl)
        if name == 'epochtime':
            ax.set_ylim(bottom=0)
        elif name in ['error']:
            ax.set_yscale('log')
            plotutil.set_log_ticks(ax.yaxis)
            ax.set_ylim(1e-3, 1)
        elif name in ['error_chi']:
            ax.set_ylim(0, 1.5)
        elif name in ['loss']:
            ax.set_yscale('log')
            ax.set_ylim(1e-3, 10)
            ax.set_yticks(10**np.arange(-3, 1.1))
            plotutil.set_log_ticks(ax.yaxis)
        else:
            ax.set_yscale('log')
        plotutil.savefig(fig,
                         os.path.join(args.dir, 'train_' + name),
                         pad_inches=0.05)
        plt.close(fig)
    plotutil.savelegend(fig, ax, os.path.join(args.dir, 'train_leg'))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir',
                        type=str,
                        default='.',
                        nargs='?',
                        help="Base directory")
    parser.add_argument('--timewidth',
                        type=int,
                        default=5,
                        help="Epoch time timewidth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    files = sorted(glob(os.path.join(args.dir, 'train_*.csv')))
    lines = [(f, re.findall('train_(\d*).csv', os.path.basename(f))[0])
             for f in files]
    if not len(lines):
        print("No train_*.csv files found in '{}'".format(args.dir))
        exit()
    plot(lines, args)
