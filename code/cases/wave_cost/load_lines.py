#!/usr/bin/env python3

import numpy as np
import re
import os
import json
from glob import glob
from util_cache import cache_to_file
from argparse import Namespace
import pickle


@cache_to_file("data.pickle")
def read_data(filesdone):
    res = dict()
    for i, fd in enumerate(filesdone):
        basedir = os.path.dirname(fd)
        if i % 100 == 0:
            print("{:}/{:} {}".format(i + 1, len(filesdone), basedir))
        with open(os.path.join(basedir, 'args.json'), 'r') as f:
            args = json.load(f)
        train = np.genfromtxt(os.path.join(basedir, 'train.csv'),
                              names=True,
                              delimiter=',')
        train = np.atleast_1d(train)
        res[basedir] = dict()
        res[basedir]['args'] = args
        res[basedir]['train'] = train
    return res


@cache_to_file("lines.pickle")
def load_lines(filesdone, error_factor=1.5, max_ntrainable=160**2):
    '''
    error_factor: multiplier for the best error to define
                  the number if significant epochs,
    '''
    data = read_data(filesdone)

    lines = Namespace()
    lines.N = []
    lines.error = []
    lines.loss = []
    lines.time_total = []
    lines.time_epoch = []
    lines.epochs = []
    lines.signif_epochs = []
    lines.signif_time = []

    info = None

    for basedir in data:
        args = data[basedir]['args']
        args = Namespace(**args)
        if info is None:
            info = Namespace()
            if args.solver == 'pinn':
                info.Nci = args.Nci
                info.Ncb = args.Ncb
            info.epochs = args.epochs
        train = data[basedir]['train']
        if args.ntrainable > max_ntrainable:
            continue
        lines.N.append(args.ntrainable)
        kerror = 'ref_du_l2'
        lines.error.append(train[kerror][-1])
        lines.loss.append(train['loss'][-1])
        lines.epochs.append(train['epoch'][-1])

        lines.time_total.append(train['tt_opt'][-1])
        if len(train['tt_opt']) > 1:
            # Time per epoch computed from the last entry in history.
            # This will exclude the first tracing of tf.function.
            last_epochs = train['epoch'][-1] - train['epoch'][-2]
            last_time = train['tt_opt'][-1] - train['tt_opt'][-2]
        else:
            last_time = train['tt_opt'][-1]
            last_epochs = train['epoch'][-1]
        if not last_epochs:
            last_time = 0
            last_epochs = 1
        lines.time_epoch.append(last_time / last_epochs)

        best = train[kerror].min()
        if args.solver == 'fd' and args.optimizer == 'newton':
            # For Newton, assume one epoch is needed since the problem in linear.
            lines.signif_epochs.append(1)
        else:
            where = np.where((train[kerror] <= best * error_factor)
                             & (train['epoch'] > 0))[0]
            if len(where) == 0:
                signif_epochs = 1
            else:
                signif_epochs = train[np.min(where)]['epoch']
            lines.signif_epochs.append(signif_epochs)
        lines.signif_time.append(lines.time_epoch[-1] *
                                    lines.signif_epochs[-1])

    for k in vars(lines):
        vars(lines)[k] = np.array(vars(lines)[k])
    return lines


def plot(lines):
    import plottools
    plottools.apply_params(plt)
    plt.rcParams.update({'font.size': 7})
    figsize = (1.7, 1.5)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.scatter(lines.N,
               lines.error,
               marker='.',
               c='k',
               s=6,
               alpha=0.25,
               edgecolor='none')
    ax.plot(lines.N, 25 * lines.N**(-1.))
    ax.set_xlabel('parameters N')
    ax.set_ylabel('error')
    ax.set_xlim(10, 1e5)
    ax.set_ylim(1e-3, 1)
    plottools.savefig(fig, 'error.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.scatter(lines.N,
               lines.time_epoch,
               marker='.',
               c='k',
               s=6,
               alpha=0.25,
               edgecolor='none')
    N = np.linspace(lines.N.min(), lines.N.max(), 100)
    ax.plot(N, N * 0.00003, ls='--')
    ax.set_xlabel('parameters N')
    ax.set_ylabel('time per evaluation [s]')
    ax.set_xlim(10, 1e5)
    ax.set_ylim(1e-3, 10)
    plottools.savefig(fig, 'time_epoch.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.scatter(lines.N,
               lines.epochs,
               marker='.',
               c='k',
               s=6,
               alpha=0.5,
               edgecolor='none')
    ax.scatter(lines.N,
               lines.signif_epochs,
               marker='.',
               c='C0',
               s=6,
               alpha=0.5,
               edgecolor='none')
    N = np.linspace(lines.N.min(), lines.N.max(), 100)
    ax.set_xlabel('parameters N')
    ax.set_ylabel('epochs for best error')
    ax.set_xlim(10, 1e5)
    ax.set_ylim(10, 1e6)
    plottools.savefig(fig, 'epochs_for_error.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.scatter(lines.N,
               lines.signif_epochs * lines.time_epoch / 3600,
               marker='.',
               c='C0',
               s=6,
               alpha=0.5,
               edgecolor='none')
    ax.set_xlabel('parameters N')
    ax.set_ylabel('execution time [s]')
    ax.set_xlim(10, 1e5)
    ax.set_ylim(1e-5, 1e1)
    plottools.savefig(fig, 'signif_time.pdf')
    plt.close(fig)


def main():
    filesdone = sorted(glob("data_*/done"))
    lines = load_lines(filesdone)


if __name__ == "__main__":
    main()
