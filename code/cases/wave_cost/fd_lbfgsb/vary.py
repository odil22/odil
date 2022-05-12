#!/usr/bin/env python3

from mpi4py import MPI
from vary_wave import run, runall
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--logN_min", type=float, default=1)
parser.add_argument("--logN_max", type=float, default=5)
parser.add_argument("--logN_samples", type=int, default=50)
parser.add_argument("--logN_skip", type=int, default=1)
parser.add_argument("--epochs", type=int, default=200000)
args = parser.parse_args()

v_N_log10 = np.linspace(args.logN_min, args.logN_max,
                        args.logN_samples)[::args.logN_skip]
v_Nx_log10 = v_N_log10 * 0.5
v_Nx = np.unique(np.round(10**v_Nx_log10).astype(int))
extra = ''
extra += '--frames  0 --plot_every 10000 --history_every 100 --report_every 100'
extra += ' --epochs {:} --optimizer lbfgsb'.format(args.epochs)
extra = extra.split()
tuples = v_Nx
func = lambda t: run(seed=0, Nx=t, extra=extra)
runall(MPI.COMM_WORLD, tuples, func)
