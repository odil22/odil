#!/usr/bin/env python3

from mpi4py import MPI
from subprocess import check_call
import itertools
import os
import numpy as np


def run(seed=None, Nx=None, extra=None, arch=None, skip_existing=True):
    '''
    seed: seed for random numbers
    Nx: grid size in t and x
    extra: list of extra arguments, list(str)
    '''
    cmd = ['./wave.py']
    outdir = "data"
    if Nx is not None:
        cmd += ['--Nx', str(Nx), '--Nt', str(Nx)]
        outdir += "_Nx{:03d}".format(Nx)
    if seed is not None:
        cmd += ['--seed', str(seed)]
        outdir += "_seed{:03d}".format(seed)
    if arch is not None:
        cmd += ['--arch', str(arch), str(arch)]
        outdir += "_arch{:03d}".format(arch)

    cmd = cmd[:1] + ['--outdir', outdir] + cmd[1:]
    if extra is not None:
        cmd += extra

    if skip_existing and os.path.isdir(outdir):
        print("skip existing " + outdir)
    else:
        print(' '.join(cmd))
        check_call(cmd)

def runall(comm, tuples, func):
    rank = comm.Get_rank()
    size = comm.Get_size()

    tuples = list(tuples)
    for t in tuples[rank::size]:
        func(t)


if __name__ == "__main__":
    v_seed = list(range(1))
    v_N_log10 = np.linspace(1, 5, 25)
    v_Nx_log10 = v_N_log10 * 0.5
    v_Nx = np.unique(np.round(10**v_Nx_log10).astype(int))
    extra = ['--frames', '0', '--plot_every', '1', '--epochs', '1']
    tuples = itertools.product(v_seed, v_Nx)
    func = lambda t: run(seed=t[0], Nx=t[1], extra=extra)
    runall(MPI.COMM_WORLD, tuples, func)
