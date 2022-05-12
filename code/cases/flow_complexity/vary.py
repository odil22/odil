#!/usr/bin/env python3

from mpi4py import MPI
from vary_ns_steady import run, runall
import itertools
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=50)
parser.add_argument("--nimp_min", type=int, default=1)
parser.add_argument("--nimp_max", type=int, default=50)
parser.add_argument("--nimp_skip", type=int, default=1)
args, unknown = parser.parse_known_args()

v_seed = list(range(args.seeds))
v_nimp = list(range(args.nimp_min, args.nimp_max, args.nimp_skip))
tuples = itertools.product(v_seed, v_nimp)

extra = ''
extra += ' --report_every 1 --history_every 1 --checkpoint_every 0'
extra += ' --plot_every 60 --frames 0 --epochs 60'
extra += ' --linsolver direct --N 64 --Re 3200 --bc 2 --imposed random --kreg 1e-3'
extra = extra.split() + unknown

func = lambda t: run(seed=t[0], nimp=t[1], extra=extra)
runall(MPI.COMM_WORLD, tuples, func)
