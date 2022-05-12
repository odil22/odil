#!/usr/bin/env python3

from mpi4py import MPI
from subprocess import check_call
import itertools
import os


def run(seed=None, N=None, Re=None, nimp=None, extra=None, skip_existing=True):
    '''
    seed: seed for random numbers
    N: grid size in x and y
    Re: Reynolds number
    nimp: number of points where the reference solution is imposed
    extra: list of extra arguments, list(str)
    '''
    cmd = ['./ns_steady.py']
    outdir = "data"
    if N is not None:
        cmd += ['--N', str(N)]
        outdir += "_N" + str(N)
    if Re is not None:
        cmd += ['--Re', str(Re)]
        outdir += "_Re" + str(Re)
    if nimp is not None:
        cmd += ['--nimp', str(nimp)]
        outdir += "_nimp{:04d}".format(nimp)
    if seed is not None:
        cmd += ['--seed', str(seed)]
        outdir += "_seed{:03d}".format(seed)

    cmd = cmd[:1] + ['--outdir', outdir] + cmd[1:]
    if extra is not None:
        cmd += extra

    if skip_existing and os.path.isdir(outdir):
        print("skip existing " + outdir)
    else:
        print(' '.join(cmd))
        check_call(cmd)
    return outdir

def runall(comm, tuples, func):
    rank = comm.Get_rank()
    size = comm.Get_size()

    tuples = list(tuples)
    for t in tuples[rank::size]:
        func(t)


if __name__ == "__main__":
    v_seed = list(range(50))
    v_nimp = list(range(1, 50))
    tuples = itertools.product(v_seed, v_nimp)
    func = lambda t: run(seed=t[0], nimp=t[1])
    runall(MPI.COMM_WORLD, tuples, func)
