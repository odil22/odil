#!/usr/bin/env python3

from collections import defaultdict
from scipy.interpolate import interp1d, RectBivariateSpline
from util_cache import cache_to_file
from util import printlog, set_log_file
from util import TIME, TIMECLEAR
from util_op import extrap_linear, extrap_quadh, extrap_quad
import argparse
import json
import math
import numpy as np
import os
import scipy.optimize
import scipy.sparse
import subprocess
import sys
import tensorflow as tf
import time
import util_op

import linsolver
import matplotlib.pyplot as plt
import optimizer
import util

g_time_start = time.time()
g_time_callback = 0.


def get_anneal_factor(epoch):
    return 0.5**(epoch / args.anneal_half) if args.anneal_half else 1


def load_fields_interp(path, domain, subdomain=None):
    '''
    Loads fields from file `path` and interpolates them to shape `domain.shape`.
    '''
    ext = os.path.splitext(path)[1]
    if ext in ['.npz']:
        d = np.load(path)
        uu = d['vx'].T
        vv = d['vy'].T
        pp = d['p'].T
    elif os.path.splitext(path)[1] in ['.pickle']:
        state = util_op.State()
        util_op.checkpoint_load(state,
                                path,
                                fields_to_load=('u', 'v', 'p'),
                                weights_to_load=[])
        uu = state.fields['u']
        vv = state.fields['v']
        pp = state.fields['p']
    else:
        raise ValueError("Unknown extension: '{}'".format(ext))

    if uu.shape != domain.shape or subdomain is not None:
        x1 = np.linspace(0, 1, uu.shape[0], endpoint=False)
        y1 = np.linspace(0, 1, uu.shape[1], endpoint=False)
        x1 += (x1[1] - x1[0]) * 0.5
        y1 += (y1[1] - y1[0]) * 0.5
        fu = RectBivariateSpline(x1, y1, uu)
        fv = RectBivariateSpline(x1, y1, vv)
        fp = RectBivariateSpline(x1, y1, pp)

        if subdomain is None:
            subdomain = [0, 1, 0, 1]
        x = np.linspace(subdomain[0],
                        subdomain[1],
                        domain.shape[0],
                        endpoint=False)
        y = np.linspace(subdomain[2],
                        subdomain[3],
                        domain.shape[1],
                        endpoint=False)
        x += (x[1] - x[0]) * 0.5
        y += (y[1] - y[0]) * 0.5
        uu = fu(x, y)
        vv = fv(x, y)
        pp = fp(x, y)
    return uu, vv, pp


def load_reference(name, domain, Re=None, N=None):
    '''
    Loads reference solution trying various sources in the following order:
        - preset solution defined by name `basename(name)` in `get_fields_preset()`
        - file `name`
        - file `name/ref_Re*_N*.npz` with given `Re` and `N`
    Returns:
        - fields, list of `ndarray`
        - name or path of loaded file
    '''
    res = get_fields_preset(os.path.basename(name), domain, args)
    if res is not None:
        return res, os.path.basename(name)
    path = name
    if not os.path.isfile(path):
        path = os.path.join(path, "ref_Re{:.0f}_N{:}.npz".format(Re, N))
    if os.path.isfile(path):
        res = load_fields_interp(path, domain, args.subdomain)
        return res, path
    else:
        return None, None


def get_fields_preset(name, domain, args):
    xx, yy = domain.cell_center_all()

    def normalize(uu, vv):
        mag = ((uu**2 + vv**2)**0.5).max()
        uu /= mag
        vv /= mag

    if name == 'uniform':
        uu = 0.2 + xx * 0
        vv = 0.5 + yy * 0
        normalize(uu, vv)
        pp = np.zeros_like(xx)
        return uu, vv, pp
    elif name == 'couette':
        qx, qy = 0.3, 0.1
        nx, ny = -qy, qx
        k = nx * xx + ny * yy
        uu = qx * k
        vv = qy * k
        normalize(uu, vv)
        pp = np.zeros_like(xx)
        return uu, vv, pp
    elif name == 'pois':  # Poiseuille.
        qx, qy = 0.9, 0.7
        nx, ny = -qy, qx
        k = nx * (xx - 0.5) + ny * (yy - 0.5)
        uu = qx * k**2
        vv = qy * k**2
        normalize(uu, vv)
        pp = np.zeros_like(xx)
        return uu, vv, pp
    return None


def operator(mod, ctx):
    global args
    dx = ctx.step('x')
    dy = ctx.step('y')
    ones = ctx.field('ones')
    zeros = ctx.field('zeros')
    ix = ctx.cell_index('x')
    iy = ctx.cell_index('y')
    nx = ctx.size('x')
    ny = ctx.size('y')
    epoch = ctx.epoch

    def stencil_var(key, freeze=False):
        'Returns: q, qxm, qxp, qym, qyp'
        st = [
            ctx.field(key, freeze=freeze),
            ctx.field(key, -1, 0, freeze=freeze),
            ctx.field(key, 1, 0, freeze=freeze),
            ctx.field(key, 0, -1, freeze=freeze),
            ctx.field(key, 0, 1, freeze=freeze)
        ]
        return st

    def central(st):
        q, qxm, qxp, qym, qyp = st
        q_x = (qxp - qxm) / (2 * dx)
        q_y = (qyp - qym) / (2 * dy)
        return q_x, q_y

    def upwind_second(st, st5, u, v):
        'Second-order upwind.'
        q, qxm, qxp, qym, qyp = st
        qxmm, qxpp, qymm, qypp = st5

        q_xm = (qxmm - 4 * qxm + 3 * q) / (2 * dx)
        q_xp = (-3 * q + 4 * qxp - qxpp) / (2 * dx)
        q_ym = (qymm - 4 * qym + 3 * q) / (2 * dy)
        q_yp = (-3 * q + 4 * qyp - qypp) / (2 * dy)

        # First order near boundaries.
        q_xm = mod.where(ix == 0, (q - qxm) / dx, q_xm)
        q_xp = mod.where(ix == nx - 1, (qxp - q) / dx, q_xp)
        q_ym = mod.where(iy == 0, (q - qym) / dy, q_ym)
        q_yp = mod.where(iy == ny - 1, (qyp - q) / dy, q_yp)

        q_x = mod.where(u > 0, q_xm, q_xp)
        q_y = mod.where(v > 0, q_ym, q_yp)
        return q_x, q_y

    def upwind_high(st, st5, u, v):
        'High-order upwind.'
        return upwind_second(st, st5, u, v)

    def upwind_low(st, u, v):
        'First-order upwind.'
        q, qxm, qxp, qym, qyp = st
        q_xm = (q - qxm) / dx
        q_xp = (qxp - q) / dx
        q_ym = (q - qym) / dy
        q_yp = (qyp - q) / dy
        q_x = mod.where(u > 0, q_xm, q_xp)
        q_y = mod.where(v > 0, q_ym, q_yp)
        return q_x, q_y

    def laplace(st):
        q, qxm, qxp, qym, qyp = st
        q_xx = (qxp - 2 * q + qxm) / dx**2
        q_yy = (qyp - 2 * q + qym) / dy**2
        q_lap = q_xx + q_yy
        return q_lap

    def laplace_split(st):
        q, qxm, qxp, qym, qyp = st
        q_xx = (qxp - 2 * q + qxm) / dx**2
        q_yy = (qyp - 2 * q + qym) / dy**2
        return q_xx, q_yy

    def apply_bc_u(st):
        'Quadratic extrapolation with boundary conditions.'
        st[1] = mod.where(ix == 0, extrap_quadh(st[2], st[0], 0), st[1])
        st[2] = mod.where(ix == nx - 1, extrap_quadh(st[1], st[0], 0), st[2])
        st[3] = mod.where(iy == 0, extrap_quadh(st[4], st[0], 0), st[3])
        st[4] = mod.where(iy == ny - 1, extrap_quadh(st[3], st[0], 1), st[4])
        return st

    def apply_bc_v(st):
        'Quadratic extrapolation with boundary conditions.'
        st[1] = mod.where(ix == 0, extrap_quadh(st[2], st[0], 0), st[1])
        st[2] = mod.where(ix == nx - 1, extrap_quadh(st[1], st[0], 0), st[2])
        st[3] = mod.where(iy == 0, extrap_quadh(st[4], st[0], 0), st[3])
        st[4] = mod.where(iy == ny - 1, extrap_quadh(st[3], st[0], 0), st[4])
        return st

    def apply_bc_extrap(st):
        'Linear extrapolation from inner cells to halo cells.'
        st[1] = mod.where(ix == 0, extrap_linear(st[2], st[0]), st[1])
        st[2] = mod.where(ix == nx - 1, extrap_linear(st[1], st[0]), st[2])
        st[3] = mod.where(iy == 0, extrap_linear(st[4], st[0]), st[3])
        st[4] = mod.where(iy == ny - 1, extrap_linear(st[3], st[0]), st[4])
        return st

    def apply_bc_npen_u(st):
        'No-penetration conditions for u.'
        st[1] = mod.where(ix == 0, extrap_quadh(st[2], st[0], 0), st[1])
        st[2] = mod.where(ix == nx - 1, extrap_quadh(st[1], st[0], 0), st[2])
        st[3] = mod.where(iy == 0, extrap_linear(st[4], st[0]), st[3])
        st[4] = mod.where(iy == ny - 1, extrap_linear(st[3], st[0]), st[4])
        return st

    def apply_bc_npen_v(st):
        'No-penetration conditions for v.'
        st[1] = mod.where(ix == 0, extrap_linear(st[2], st[0]), st[1])
        st[2] = mod.where(ix == nx - 1, extrap_linear(st[1], st[0]), st[2])
        st[3] = mod.where(iy == 0, extrap_quadh(st[4], st[0], 0), st[3])
        st[4] = mod.where(iy == ny - 1, extrap_quadh(st[3], st[0], 0), st[4])
        return st

    def stencil(q):
        'Returns: q, qxm, qxp, qym, qyp.'
        st = [None] * 5
        st[0] = q
        st[1] = mod.roll(st[0], shift=1, axis=0)
        st[2] = mod.roll(st[0], shift=-1, axis=0)
        st[3] = mod.roll(st[0], shift=1, axis=1)
        st[4] = mod.roll(st[0], shift=-1, axis=1)
        return st

    def stencil5(st):
        'Returns: qxmm, qxpp, qymm, qypp.'
        st5 = [None] * 4
        st5[0] = mod.roll(st[1], shift=1, axis=0)
        st5[1] = mod.roll(st[2], shift=-1, axis=0)
        st5[2] = mod.roll(st[3], shift=1, axis=1)
        st5[3] = mod.roll(st[4], shift=-1, axis=1)
        st5[0] = mod.where(ix == 0, st[1], st5[0])
        st5[1] = mod.where(ix == nx - 1, st[2], st5[1])
        st5[2] = mod.where(iy == 0, st[3], st5[2])
        st5[3] = mod.where(iy == ny - 1, st[4], st5[3])
        return st5

    def upwind_mix(vhf, vl, vlf):
        return (hf + l - lf for hf, l, lf in zip(vhf, vl, vlf))

    u_st = stencil_var('u')
    uf_st = stencil_var('u', freeze=True)
    if args.bc == 0:
        apply_bc_u(u_st)
        apply_bc_u(uf_st)
    elif args.bc == 1:
        apply_bc_npen_u(u_st)
        apply_bc_npen_u(uf_st)
    elif args.bc == 2:
        apply_bc_extrap(u_st)
        apply_bc_extrap(uf_st)
    else:
        raise ValueError("Unknown bc={:}".format(args.bc))
    uf_st5 = stencil5(uf_st)
    uf = uf_st[0]

    v_st = stencil_var('v')
    vf_st = stencil_var('v', freeze=True)
    if args.bc == 0:
        apply_bc_v(v_st)
        apply_bc_v(vf_st)
    elif args.bc == 1:
        apply_bc_npen_v(v_st)
        apply_bc_npen_v(vf_st)
    elif args.bc == 2:
        apply_bc_extrap(v_st)
        apply_bc_extrap(vf_st)
    else:
        raise ValueError("Unknown bc={:}".format(args.bc))
    vf_st5 = stencil5(vf_st)
    vf = vf_st[0]

    p_st = stencil_var('p')
    apply_bc_extrap(p_st)

    u_x, u_y = upwind_mix(upwind_high(uf_st, uf_st5, uf, vf),
                          upwind_low(u_st, uf, vf), upwind_low(uf_st, uf, vf))
    v_x, v_y = upwind_mix(upwind_high(vf_st, vf_st5, uf, vf),
                          upwind_low(v_st, uf, vf), upwind_low(vf_st, uf, vf))
    p_xc, p_yc = central(p_st)
    u_lap = laplace(u_st)
    v_lap = laplace(v_st)
    mu = args.mu

    if args.fit_mu:
        k = ctx.neural_net('k')
        u = u_st[0]
        v = v_st[0]
        mu = mu * (1 + (abs(k(u**2 + v**2)) - 1) * args.fit_mu)

    # Momentum equations.
    fu = uf * u_x + vf * u_y - mu * u_lap + p_xc
    fv = uf * v_x + vf * v_y - mu * v_lap + p_yc

    if args.rhiechow:
        # Rhie-Chow correction to remove oscillations.
        '''
        Diagonal part of momentum equations:
            u = - 1/g p_x
            v = - 1/g p_y
        '''
        g = mod.abs(uf) / dx + mod.abs(vf) / dy + mu * (2 / dx**2 + 2 / dy**2)
        u, uxm, uxp, _, _ = u_st
        v, _, _, vym, vyp = v_st
        pf_st = stencil_var('p', freeze=True)
        apply_bc_extrap(pf_st)
        pf, pfxm, pfxp, pfym, pfyp = pf_st
        pfxmm, pfxpp, pfymm, pfypp = stencil5(pf_st)
        p, pxm, pxp, pym, pyp = p_st

        g, gxm, gxp, gym, gyp = stencil(g)

        qxm = (u + uxm) * 0.5 + \
            (1 / gxm * (pf - pfxmm) / (2 * dx)) * 0.5 + \
            (1 / g * (pfxp - pfxm) / (2 * dx)) * 0.5 - \
            (1 / gxm + 1 / g) * 0.5 * (p - pxm) / dx
        qxp = (u + uxp) * 0.5 + \
            (1 / gxp * (pfxpp - pf) / (2 * dx)) * 0.5 + \
            (1 / g * (pfxp - pfxm) / (2 * dx)) * 0.5 - \
            (1 / gxp + 1 / g) * 0.5 * (pxp - p) / dx
        qym = (v + vym) * 0.5 + \
            (1 / gym * (pf - pfymm) / (2 * dy)) * 0.5 + \
            (1 / g * (pfyp - pfym) / (2 * dy)) * 0.5 - \
            (1 / gym + 1 / g) * 0.5 * (p - pym) / dy
        qyp = (v + vyp) * 0.5 + \
            (1 / gyp * (pfypp - pf) / (2 * dy)) * 0.5 + \
            (1 / g * (pfyp - pfym) / (2 * dy)) * 0.5 - \
            (1 / gyp + 1 / g) * 0.5 * (pyp - p) / dy

        if args.bc == 2:
            'Keep central fluxes for extrapolation conditions'
            qxm = mod.where(ix == 0, (u + uxm) * 0.5, qxm)
            qxp = mod.where(ix == nx - 1, (u + uxp) * 0.5, qxp)
            qym = mod.where(iy == 0, (v + vym) * 0.5, qym)
            qyp = mod.where(iy == ny - 1, (v + vyp) * 0.5, qyp)
        else:
            qxm = mod.where(ix == 0, zeros, qxm)
            qxp = mod.where(ix == nx - 1, zeros, qxp)
            qym = mod.where(iy == 0, zeros, qym)
            qyp = mod.where(iy == ny - 1, zeros, qyp)

        fp = (qxp - qxm) / dx + (qyp - qym) / dy
    else:
        # Continuity equation without correction,
        # will produce oscillatory pressure.
        u_xc, _ = central(u_st)
        _, v_yc = central(v_st)
        fp = u_xc + v_yc

    res = [fu, fv, fp]

    global mask_imp, u_imp, v_imp, n_imp
    if n_imp:
        u = u_st[0]
        v = v_st[0]
        # Rescale weight to the total number of points.
        coeff = args.alpha * np.prod(domain.shape) / n_imp
        fuimp = mask_imp * (u - u_imp) * coeff
        fvimp = mask_imp * (v - v_imp) * coeff
        res += [fuimp, fvimp]

    if args.kreg:
        anneal_factor = get_anneal_factor(ctx.epoch)
        u_xx, u_yy = laplace_split(u_st)
        v_xx, v_yy = laplace_split(v_st)
        k = args.kreg * anneal_factor
        res += [u_xx * k, u_yy * k, v_xx * k, v_yy * k]

    def apply_bc_psi(st):
        'Quadratic extrapolation with boundary conditions.'
        st[1] = mod.where(ix == 0, extrap_quadh(st[2], st[0], 0), st[1])
        st[2] = mod.where(ix == nx - 1, extrap_quadh(st[1], st[0], 0), st[2])
        st[3] = mod.where(iy == 0, extrap_quadh(st[4], st[0], 0), st[3])
        st[4] = mod.where(iy == ny - 1, extrap_quadh(st[3], st[0], 0), st[4])
        return st

    if args.plot_stream:
        x = ctx.cell_center('x')
        y = ctx.cell_center('y')
        psi_st = stencil_var('psi')
        apply_bc_psi(psi_st)
        psi, _, _, _, _ = psi_st
        uf_x, uf_y = central(uf_st)
        vf_x, vf_y = central(vf_st)
        omega = vf_x - uf_y
        fpsi = laplace(psi_st) + omega
        res += [fpsi]

    return res


def get_indices_imposed(i_diag):
    if args.imposed == 'edge':
        i_imposed = np.hstack((
            i_diag[0, :],
            i_diag[-1, :],
            i_diag[:, 0],
            i_diag[:, -1],
        )).flatten()
    elif args.imposed == 'edge2':
        i_imposed = np.hstack((
            i_diag[0, :],
            i_diag[-1, :],
            i_diag[:, 0],
            i_diag[:, -1],
            i_diag[1, :],
            i_diag[-2, :],
            i_diag[:, 1],
            i_diag[:, -2],
        )).flatten()
    elif args.imposed == 'none':
        i_imposed = []
    elif args.imposed == 'all':
        i_imposed = i_diag
    elif args.imposed == 'random':
        i_imposed = i_diag.flatten()
        i_imposed = np.random.choice(i_imposed, size=args.nimp, replace=False)
    elif args.imposed == 'random_edge':
        i_imposed = np.hstack((
            i_diag[0, :],
            i_diag[-1, :],
            i_diag[:, 0],
            i_diag[:, -1],
        )).flatten()
        i_imposed = np.random.choice(i_imposed, size=args.nimp, replace=False)
    else:
        raise ValueError("Unknown imposed=" + args.imposed)
    return i_imposed


def get_mask_imposed(domain):
    size = np.prod(domain.shape)
    Nf = len(domain.fieldnames)
    row = range(size)
    i_diag = np.reshape(row, domain.shape)
    i_imposed = get_indices_imposed(i_diag)
    i_imposed = np.unique(i_imposed)
    res = np.zeros(size)
    xx = []
    yy = []
    if len(i_imposed):
        res[i_imposed] = 1
        xx = domain.cell_center_by_dim(0).flatten()
        yy = domain.cell_center_by_dim(1).flatten()
        xx = xx[i_imposed]
        yy = yy[i_imposed]
    res = res.reshape(domain.shape)
    return res, xx, yy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Re',
                        type=float,
                        default=3200,
                        help="Reynolds number")
    parser.add_argument('--alpha',
                        type=float,
                        default=1,
                        help="Loss weight for imposed values")
    parser.add_argument('--anneal_half',
                        type=float,
                        default=0,
                        help="Number of epochs before halving "
                        "of regularization factors")
    parser.add_argument('--kreg',
                        type=float,
                        default=0,
                        help="Laplacian regularization factor")
    parser.add_argument('--N', type=int, default=64, help="Grid size")
    parser.add_argument('--nimp',
                        type=int,
                        default=32,
                        help="Number of points for imposed=random")
    parser.add_argument('--ref_path',
                        type=str,
                        default="ref_cavity",
                        help="Path to directory with reference solution")
    parser.add_argument('--load_weights',
                        type=str,
                        help="Path to checkpoint with initial weights")
    parser.add_argument('--load_fields',
                        type=str,
                        help="Path to checkpoint with initial fields")
    parser.add_argument('--frozen_weights',
                        type=str,
                        nargs='?',
                        default=[],
                        help="List of weights to freeze "
                        "(omit gradients and updates)")
    parser.add_argument('--subdomain',
                        type=float,
                        nargs=4,
                        default=None,
                        help="Subdomain for reference (x0, x1, y0, y1)")
    parser.add_argument('--rhiechow',
                        type=int,
                        default=1,
                        help="Use Rhie-Chow correction to remove "
                        "pressure oscillations")
    parser.add_argument('--fit_mu',
                        type=float,
                        default=0,
                        help="Factor to fit effective viscosity")
    parser.add_argument('--plot_stream',
                        type=float,
                        default=0,
                        help="Plot stream function")
    parser.add_argument('--imposed',
                        type=str,
                        choices=('all', 'edge', 'random', 'random_edge',
                                 'none', 'edge2'),
                        default='none',
                        help="Set of points for imposed solution")
    parser.add_argument('--bc',
                        type=int,
                        default=0,
                        help="Boundary conditions "
                        "(0: noslip, 1: no-penetraion, 2: extrapolation)")
    parser.add_argument('--ref_N',
                        type=int,
                        default=128,
                        help="Reference solution grid size. "
                        "Defaults to N if zero")
    util.add_arguments(parser)
    linsolver.add_arguments(parser)
    return parser.parse_args()


def plot_neural_1d(weights,
                   path,
                   title=None,
                   dpi=300,
                   ymin=None,
                   ymax=None,
                   xlabel=None,
                   ylabel=None,
                   xmin=0,
                   xmax=1):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x = np.linspace(xmin, xmax, 100)
    if args.fit_mu:  # XXX adhoc
        y = util_op.eval_neural_net(weights, x).numpy()
    else:
        y = 0 * x
        ymin = -1
        umax = 1
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.plot(x, y, 'r-', linewidth=1)
    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

def get_levels_ghia():
    # Ghia1982, Table III
    levels = []
    # Contour letters.
    levels += [
        -1e-10, -1e-7, -1e-5, -1e-4, -0.01, -0.03, -0.05, -0.07, -0.09, -0.1,
        -0.11, -0.115, -0.1175
    ]
    # Contour numbers.
    levels += [
        1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 1.5e-3, 3e-3
    ]
    levels = sorted(levels)
    return levels


def plot_stream(domain,
                psi,
                path,
                dpi=300,
                transparent=True,
                levels=None):
    fig, ax = plt.subplots(figsize=(2, 2))
    extent = [
        domain.lower[0], domain.upper[0], domain.lower[1], domain.upper[1]
    ]
    ax.contour(psi.T,
               levels=levels,
               extent=extent,
               origin='lower',
               linewidths=0.5,
               linestyles='-',
               colors='k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    fig.savefig(path,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0.01,
                transparent=transparent)
    plt.close(fig)


def plot(state, epoch, frame):
    global exact_uu, exact_vv
    global x_imp, y_imp
    title0 = "u epoch={:05d}".format(epoch) if args.plot_title else None
    title1 = "v epoch={:05d}".format(epoch) if args.plot_title else None
    title2 = "p epoch={:05d}".format(epoch) if args.plot_title else None
    title3 = "k epoch={:05d}".format(epoch) if args.plot_title else None
    path0 = "u_{:05d}.pdf".format(frame)
    path1 = "v_{:05d}.pdf".format(frame)
    path2 = "p_{:05d}.pdf".format(frame)
    path3 = "k_{:05d}.pdf".format(frame)
    path4 = "psi_{:05d}.pdf".format(frame)
    printlog(path0, path1, path2)
    uu, vv, pp, *_ = [
        np.array(state.fields[name]) for name in domain.fieldnames
    ]
    umax = max(abs(np.max(exact_uu)), abs(np.min(exact_uu)))
    if not umax:
        umax = max(abs(np.max(uu)), abs(np.min(uu)))
    umax = max(0.01, umax)
    util.plot(domain,
              exact_uu,
              uu,
              path=path0,
              title=title0,
              cmap='RdBu_r',
              nslices=5,
              x_imp=x_imp,
              y_imp=y_imp,
              umin=-umax,
              umax=umax)
    umax = max(abs(np.max(exact_vv)), abs(np.min(exact_vv)))
    if not umax:
        umax = max(abs(np.max(vv)), abs(np.min(vv)))
    umax = max(0.01, umax)
    util.plot(domain,
              exact_vv,
              vv,
              path=path1,
              title=title1,
              cmap='RdBu_r',
              nslices=5,
              x_imp=x_imp,
              y_imp=y_imp,
              umin=-umax,
              umax=umax)
    pp = pp - pp.mean()

    if args.plot_stream:
        psi = np.array(state.fields['psi'])
        plot_stream(domain, psi, path4, levels=get_levels_ghia())

    umin, umax = np.quantile(exact_pp, 0.001), np.quantile(exact_pp, 0.999)
    if umin == umax:
        umin, umax = np.quantile(pp, 0.001), np.quantile(pp, 0.999)
    if umax - umin < 2e-3:
        umin = -1e-3
        umax = 1e-3
    util.plot(domain,
              exact_pp - np.mean(exact_pp),
              pp,
              path=path2,
              title=title2,
              nslices=5,
              umin=umin,
              umax=umax)
    if args.fit_mu:
        plot_neural_1d(state.weights['k'],
                       path3,
                       title=title3,
                       xlabel='u',
                       ylabel='k')
    else:
        path3 = None
    if args.montage:
        cmd = [
            "montage", "-density", "300", "-geometry", "+0+0", path0, path1, path2,
            path3, "tmp.jpg".format(epoch)
        ]
        cmd = [v for v in cmd if v is not None]
        printlog(' '.join(cmd))
        subprocess.run(cmd, check=True)
        cmd = ["mv", "tmp.jpg", "all_{:05d}.jpg".format(frame)]
        printlog(' '.join(cmd))
        subprocess.run(cmd, check=True)


def callback(packed, epoch, dhistory=None, opt=None, loss_grad=None):
    tstart = time.time()
    global frame, csv, csv_empty, history, packed_prev
    global g_time_callback, g_time_start

    def printrep(m=''):
        if epoch % args.report_every == 0:
            printlog(m)

    if packed_prev is None:
        packed_prev = packed

    state = problem.unpack_state(packed)
    dstate = problem.unpack_state(packed - packed_prev)
    packed_prev = np.copy(packed)

    printrep("epoch={:05d}".format(epoch))
    if epoch % args.plot_every == 0 and (epoch or args.frames):
        plot(state, epoch, frame)
        frame += 1
    if args.checkpoint_every and epoch % args.checkpoint_every == 0:
        path = "state_{:06d}.pickle".format(epoch)
        printlog(path)
        util_op.checkpoint_save(state, path)

    timer = util.Timer()
    timer.push("solve")
    timer.pop()
    problem.timer_total.append(timer)
    problem.timer_last.append(timer)

    printrep("T last: " + ', '.join([
        '{}={:7.3f}'.format(k, t)
        for k, t in problem.timer_last.counters.items()
    ]))
    printrep("T  all: " + ', '.join([
        '{}={:7.3f}'.format(k, t) for k, t in problem.timer_total.counters.items()
    ]))

    du, dv, dp, *_ = [
        np.array(dstate.fields[name]) for name in domain.fieldnames
    ]
    uu, vv, pp, *_ = [
        np.array(state.fields[name]) for name in domain.fieldnames
    ]
    printrep("u={:.05g} du={:.05g}".format(np.max(np.abs(uu)),
                                           np.max(np.abs(du))))
    printrep("v={:.05g} dv={:.05g}".format(np.max(np.abs(vv)),
                                           np.max(np.abs(dv))))
    printrep("p={:.05g} dp={:.05g}".format(np.max(np.abs(pp)),
                                           np.max(np.abs(dp))))

    history['epoch'].append(epoch)
    history['du'].append(np.max(np.abs(du)))
    history['dv'].append(np.max(np.abs(dv)))
    history['dp'].append(np.max(np.abs(dp)))
    history['anneal'].append(get_anneal_factor(epoch))
    history['t_solve'].append(problem.timer_last.counters['solve'])
    history['t_grad'].append(problem.timer_last.counters.get('eval_grad', 0))
    history['t_sparse_fields'].append(
        problem.timer_last.counters.get('sparse_fields', 0))
    history['t_sparse_weights'].append(
        problem.timer_last.counters.get('sparse_weights', 0))

    history['ref_du_linf'].append(np.max(abs(exact_uu - uu)))
    history['ref_du_l1'].append(np.mean(abs(exact_uu - uu)))
    history['ref_du_l2'].append(np.mean((exact_uu - uu)**2)**0.5)
    history['ref_dv_linf'].append(np.max(abs(exact_vv - vv)))
    history['ref_dv_l1'].append(np.mean(abs(exact_vv - vv)))
    history['ref_dv_l2'].append(np.mean((exact_vv - vv)**2)**0.5)
    history['ref_dp_linf'].append(np.max(abs(exact_pp - pp)))
    history['ref_dp_l1'].append(np.mean(abs(exact_pp - pp)))
    history['ref_dp_l2'].append(np.mean((exact_pp - pp)**2)**0.5)

    g_time_callback += time.time() - tstart

    history['tt_opt'].append(time.time() - g_time_start - g_time_callback)
    history['tt_callback'].append(g_time_callback)

    if 'k' not in args.frozen_weights:
        ww = np.hstack([np.reshape(w, -1) for w in state.weights['k']])
        dw = np.hstack([np.reshape(w, -1) for w in dstate.weights['k']])
        printrep("w={:.05g} dw={:.05g}".format(np.max(np.abs(ww)),
                                               np.max(np.abs(dw))))
        history['dw'].append(np.max(np.abs(dw)))
    printrep()

    if (epoch % args.history_every == 0
            or epoch < args.history_full) and csv is not None:
        keys = list(history)
        if csv_empty:
            csv.write(','.join(keys) + '\n')
            csv_empty = False
        row = [history[key][-1] for key in keys]
        line = ','.join(map(str, row))
        csv.write(line + '\n')
        csv.flush()


def main():
    global csv, csv_empty, packed_prev, args, problem, domain, frame
    global history, nhistory
    global exact_uu, exact_vv, exact_pp
    global x_imp, y_imp, n_imp, u_imp, v_imp, mask_imp

    args = parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # Replace with absolute paths before changing current directory.
    if args.load_fields is not None:
        args.load_fields = os.path.abspath(args.load_fields)
    if args.load_weights is not None:
        args.load_weights = os.path.abspath(args.load_weights)
    if args.ref_path is not None:
        args.ref_path = os.path.abspath(args.ref_path)

    # Change current directory to output directory.
    os.chdir(outdir)
    set_log_file(open("train.log", 'w'))

    csv = None
    csv_empty = True
    if args.history_every:
        csv = open('train.csv', 'w')
    history = defaultdict(lambda: [])
    packed_prev = None

    # Update arguments.
    args.mu = 1 / args.Re
    args.plot_every *= args.every_factor
    args.history_every *= args.every_factor
    args.report_every *= args.every_factor
    args.checkpoint_every *= args.every_factor
    if args.epochs is None:
        args.epochs = args.frames * args.plot_every
    ref_N = args.ref_N
    if not ref_N:
        ref_N = args.N

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    if not args.fit_mu:
        args.frozen_weights.append('k')

    neuralnets = {'k': (1, 10, 1)} if args.fit_mu else {}
    fieldnames = ['u', 'v', 'p']
    if args.plot_stream:
        fieldnames += ['psi']
    domain = util_op.Domain(ndim=2,
                            shape=(args.N, args.N),
                            dtype=np.float64,
                            varnames=('x', 'y'),
                            fieldnames=fieldnames,
                            neuralnets=neuralnets,
                            frozen_weights=args.frozen_weights)

    # Load reference solution.
    exact_uvp, path = load_reference(args.ref_path,
                                     domain,
                                     N=args.ref_N,
                                     Re=args.Re)

    printlog(' '.join(sys.argv))

    if exact_uvp is not None:
        printlog("Loading reference solution from '{}'".format(path))
    else:
        printlog("Can't load from '{}'. Setting to zeros".format(path))
        exact_uvp = [np.zeros(domain.shape) for _ in domain.fieldnames]

    exact_uu, exact_vv, exact_pp = exact_uvp[:3]

    mask_imp, x_imp, y_imp = get_mask_imposed(domain)
    n_imp = len(x_imp)
    u_imp, v_imp = exact_uvp[:2]

    if n_imp:
        with open("imposed.csv", 'w') as f:
            f.write('x,y\n')
            for i in range(n_imp):
                f.write('{:},{:}\n'.format(x_imp[i], y_imp[i]))

    Nf = len(domain.fieldnames)
    N = np.prod(domain.shape)

    shape = domain.shape
    wsize = domain.aweights_size()
    problem = util_op.Problem(operator, domain)

    state = util_op.State()

    if args.load_fields is not None:
        printlog("Loading initial fields from '{}'".format(args.load_fields))
        util_op.checkpoint_load(state,
                                args.load_fields,
                                fields_to_load=domain.fieldnames,
                                weights_to_load=[])

    if args.load_weights is not None:
        printlog("Loading initial weights from '{}'".format(args.load_weights))
        util_op.checkpoint_load(state,
                                args.load_weights,
                                fields_to_load=[],
                                weights_to_load=domain.neuralnets.keys())

    problem.init_missing(state)
    frame = 0

    if args.optimizer == 'newton':
        util.optimize_newton(args, problem, state, callback)
    else:
        util.optimize_opt(args, args.optimizer, problem, state, callback)

    with open('done', 'w') as f:
        pass


if __name__ == "__main__":
    main()
