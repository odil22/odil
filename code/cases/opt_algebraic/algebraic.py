#!/usr/bin/env python3

'''
Solves equation `u = 0`, `v = 0` with different iterative methods.
Parts depending on the equation are marked with XXX EQN
'''

import plottools
import matplotlib.pyplot as plt
plottools.apply_params(plt)
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'lines.linewidth': 1})
import numpy as np
from argparse import Namespace

dtype = np.float64


def printlog(*m):
    print(*m)

def equation(x, y): # XXX EQN
    '''
    u = x - y * y + x * y
    v = x * x + y - 2 + np.sin(y)
    ux = 1 + y
    uy = -2 * y + x
    vx = 2 * x
    vy = 1 + np.cos(y)
    '''
    u = x - y * y + x * y
    v = x - 2
    ux = 1 - y
    uy = -2 * y + x
    vx = 1
    vy = 0
    return dtype(u), dtype(v), dtype(ux), dtype(uy), dtype(vx), dtype(vy)


def history_init():
    history = Namespace()
    history.i = []
    history.u = []
    history.v = []
    history.error_x = []
    history.error_y = []
    return history

def history_finalize(history):
    history.i = np.array(history.i)
    history.u = np.array(history.u)
    history.v = np.array(history.v)
    history.error_x = np.array(history.error_x)
    history.error_y = np.array(history.error_y)
    history.loss = history.u ** 2 + history.v ** 2
    history.error = (history.error_x ** 2 + history.error_y ** 2) ** 0.5


def solve_newton(x, y, eta=0.1, niters=50):
    x = dtype(x)
    y = dtype(y)
    eta = dtype(eta)
    printlog("Solving with Newton")
    xx = [x]
    yy = [y]
    history = history_init()

    for i in range(niters):
        u, v, ux, uy, vx, vy = equation(x, y)

        history.i.append(i)
        history.u.append(u)
        history.v.append(v)
        history.error_x.append(x - exact_x)
        history.error_y.append(y - exact_y)

        a = np.array([[ux, uy], [vx, vy]])
        ainv = np.linalg.inv(a)
        dx, dy = -ainv.dot([u, v])
        x += eta * dx
        y += eta * dy
        printlog('i={:} u={:.5e} v={:.5e} x={:.5f} y={:.5f}'.format(
            i, u, v, x, y))
        xx.append(x)
        yy.append(y)

    history_finalize(history)
    return xx, yy, history


def solve_grad(x, y, eta=0.1, niters=50):
    x = dtype(x)
    y = dtype(y)
    eta = dtype(eta)
    printlog("Solving with Gradient Descent")
    xx = [x]
    yy = [y]
    history = history_init()

    for i in range(niters):
        u, v, ux, uy, vx, vy = equation(x, y)

        history.i.append(i)
        history.u.append(u)
        history.v.append(v)
        history.error_x.append(x - exact_x)
        history.error_y.append(y - exact_y)

        a = np.array([[ux, uy], [vx, vy]])
        dx, dy = -a.T.dot([u, v])
        x += eta * dx
        y += eta * dy
        printlog('i={:} u={:.5e} v={:.5e} x={:.5f} y={:.5f}'.format(
            i, u, v, x, y))
        xx.append(x)
        yy.append(y)

    history_finalize(history)
    return xx, yy, history


x0 = 1
y0 = 1
exact_x = 2. # XXX EQN
exact_y = 1 + 3 ** 0.5 # XXX EQN
figsize = (1.7, 1.5)

fig, ax = plt.subplots(figsize=figsize)
hfig, hax = plt.subplots(figsize=figsize)

ax.set_xlim(0.9, 3.1)
ax.set_ylim(0.9, 3.1)

hax.set_xlim(-2, 32)
hax.set_xticks([0, 10, 20, 30])
hax.set_ylim(1e-8, 10)

ax.scatter(x0, y0, c='k', marker='s', zorder=-1, label='initial')

niters = 30

eta=1.
xx, yy, history = solve_newton(x0, y0, eta=eta, niters=niters)
label = 'Newton $\eta$={:}'.format(eta)
l, = ax.plot(xx, yy, marker='.', c='C1', label=label)
ax.scatter(xx[-1], yy[-1], c=l.get_color(), marker='o', zorder=-1)
hax.plot(history.i, history.error, c=l.get_color(), label=label, marker='.')

eta=0.20
xx, yy, history = solve_grad(x0, y0, eta=eta, niters=niters)
label = 'GD $\eta$={:}'.format(eta)
l, = ax.plot(xx, yy, marker='.', c='C0', label=label)
ax.scatter(xx[-1], yy[-1], c=l.get_color(), marker='o', zorder=-1)
hax.plot(history.i, history.error, c=l.get_color(), label=label, marker='.')

ax.set_xlabel('x')
ax.set_ylabel('y')

hax.set_yscale('log')
hax.set_xlabel('epoch')
hax.set_ylabel('error')

plottools.savefig(fig,"traj.pdf")
plottools.savefig(hfig, "error.pdf")
plottools.savelegend(fig, ax, 'legend.pdf')
