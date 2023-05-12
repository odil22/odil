#!/usr/bin/env python3

from mpi4py import MPI
from subprocess import check_call
import itertools
import os
import numpy as np
import argparse


def run(seed=None, extra=None, skip_existing=True, outdir=''):
    '''
    seed: seed for random numbers
    extra: list of extra arguments, list(str)
    '''
    cmd = ['./wave.py']
    outdirloc = os.path.join(outdir, "out")
    if seed is not None:
        cmd += ['--seed', str(seed)]
        outdirloc += "_seed{:03d}".format(seed)

    cmd = cmd[:1] + ['--outdir', outdirloc] + cmd[1:]
    if extra is not None:
        cmd += extra

    if skip_existing and os.path.isdir(outdirloc):
        print("skip existing " + outdirloc)
    else:
        print(' '.join(cmd))
        env = dict(os.environ, NOWARN='1', OMP_NUM_THREADS='1')
        check_call(cmd, env=env)


def runall(comm, tuples, func):
    rank = comm.Get_rank()
    size = comm.Get_size()

    tuples = list(tuples)
    for t in tuples[rank::size]:
        func(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra', type=str, default='', help="Extra arguments")
    parser.add_argument('--outdir', type=str, help="Global output directory")
    args = parser.parse_args()

    v_seed = list(range(100))
    extra = ''
    extra += ' --report_every 1000 --history_every 10'
    extra += ' --plot_every 10000 --solver pinn --frames 0 --montage 0'
    extra += ' --optimizer lbfgsb --epochs 5000 --ref nosymm'
    extra += ' --arch 25 25 --Nci 512 --Ncb 256'
    extra += ' ' + args.extra
    extra = extra.split()

    func = lambda x: run(seed=x, extra=extra, outdir=args.outdir)
    runall(MPI.COMM_WORLD, v_seed, func)

