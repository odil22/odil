#!/usr/bin/env python3

'''
Solves a one-dimensional finite difference equation with boundary conditions
    u[0] = 1
    u[i] - u[i-1] = 0
'''

import plottools
import matplotlib.pyplot as plt
plottools.apply_params(plt)
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'lines.linewidth': 1})
import numpy as np
from argparse import Namespace

Nx = 5
eta = 0.9

def grad(u):
    um = np.roll(u, 1)
    up = np.roll(u, -1)
    g = -um + 2 * u - up
    g[0] = -u[1] + 2 * u[0] - 1
    g[-1] = u[-1] - u[-2]
    g *= 0.5
    return g

figsize = (1.7, 1.5)
fig, ax = plt.subplots(figsize=figsize)
hfig, hax = plt.subplots(figsize=figsize)

exact_u = np.ones(Nx)
u = np.zeros(Nx)
i = np.arange(1, Nx + 1)

history_n = []
history_error = []
nplot = [0, 1, 2, 3, 10, 30]
for n in range(max(nplot) + 1):
    if n in nplot:
        ax.plot(i, u, label='{:}'.format(n), marker='.', )
    history_n.append(n)
    history_error.append(np.mean((u - exact_u)**2)**0.5)
    u -= eta * grad(u)

ax.set_xlabel('grid point')
ax.set_ylabel('solution')
ax.set_xlim(0.5, Nx + 0.5)
ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 0.5, 1])
plottools.adjust_ticks(ax, autoscale=False, ky=0.5)
ax.set_xticks([1, 2, 3, 4, 5])

hax.set_yscale('log')
hax.plot(history_n, history_error, marker='.')
hax.set_xlim(-2, 32)
hax.set_ylim(0.06, 1.5)
hax.set_xticks([0, 10, 20, 30])
hax.set_yticklabels([], minor=True)
hax.set_xlabel('epoch')
hax.set_ylabel('error')

plottools.savefig(fig, 'field.pdf')
plottools.savefig(hfig, "error.pdf")
plottools.savelegend(fig, ax, 'legend.pdf')
