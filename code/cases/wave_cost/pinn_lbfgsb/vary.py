#!/usr/bin/env python3

from mpi4py import MPI
from vary_wave import run, runall
import itertools
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=25)
parser.add_argument("--logN_min", type=float, default=1)
parser.add_argument("--logN_max", type=float, default=5)
parser.add_argument("--logN_samples", type=int, default=50)
parser.add_argument("--logN_skip", type=int, default=1)
parser.add_argument("--epochs", type=int, default=80000)
parser.add_argument("--Nci", type=int, default=8192)
parser.add_argument("--Ncb", type=int, default=256)
args = parser.parse_args()

v_seed = list(range(args.seeds))
v_size_log10 = np.linspace(args.logN_min, args.logN_max,
                           args.logN_samples)[::args.logN_skip]
v_arch_log10 = v_size_log10 * 0.5
v_arch = np.unique(np.round(10**v_arch_log10).astype(int))
tuples = itertools.product(v_seed, v_arch)

extra = ''
extra += ' --report_every 100 --history_every 100'
extra += ' --plot_every 4000 --solver pinn --frames 0'
extra += ' --optimizer lbfgsb --epochs {:}'.format(args.epochs)
extra += ' --Nci {:} --Ncb {:}'.format(args.Nci, args.Ncb)
extra = extra.split()

func = lambda t: run(seed=t[0], arch=t[1], extra=extra)
runall(MPI.COMM_WORLD, tuples, func)
