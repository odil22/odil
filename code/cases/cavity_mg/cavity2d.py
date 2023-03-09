#!/usr/bin/env python3

from collections import defaultdict
from scipy.interpolate import interp1d, RectBivariateSpline
import argparse
import json
import math
import numpy as np
import os
import scipy.optimize
import scipy.sparse
import subprocess
import sys
import time

import odil
from odil import tf

import matplotlib.pyplot as plt
from functools import partial

g_time_start = time.time()
printlog = odil.util.printlog


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
        state = odil.util_op.State()
        odil.util_op.checkpoint_load(state,
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
        - preset solution defined by name `name` in `get_fields_preset()`
        - file `name`
        - file `name/ref_Re*_N*.npz` with given `Re` and `N`
    Returns:
        - fields, list of `ndarray`
        - name or path of loaded file
    '''
    path = name
    if os.path.isfile(path):
        res = load_fields_interp(path, domain)
        return res, path
    else:
        return None, None


def interp_field(u, rep=1, mod=None):
    if rep == 0:
        return u
    else:
        nx, ny = u.shape
        dim = len(u.shape)
        u00 = u
        u10 = sum(mod.roll(u, d, range(dim))
                  for d in [(0, 0), (-1, 0)]) * 0.5**(dim - 1)
        u01 = sum(mod.roll(u, d, range(dim))
                  for d in [(0, 0), (0, -1)]) * 0.5**(dim - 1)
        u11 = sum(
            mod.roll(u, d, range(dim))
            for d in [(0, 0), (0, -1), (-1, 0), (-1, -1)]) * 0.5**dim
        uu = [u00, u01, u10, u11]
        u = mod.batch_to_space(uu, block_shape=[2] * dim,
                               crops=[[0, 1]] * dim)[0]
        return interp_field(u, rep - 1, mod=mod)


def weights_to_fields(uw, nn, mod):
    uu = []
    s = 0
    for n in nn:
        uu.append(mod.reshape(uw[s:s + n**2], [n, n]))
        s += n**2

    return [
        interp_field(u, i, mod=mod) for i, u in enumerate(uu)
    ]


def weights_to_field(uw, nn, mod):
    return sum(weights_to_fields(uw, nn, mod=mod))


def operator_ns(mod, ctx, args=None):
    global problem
    dx = ctx.step('x')
    dy = ctx.step('y')
    x = ctx.cell_center('x')
    y = ctx.cell_center('y')
    ones = ctx.field('ones')
    zeros = ctx.field('zeros')
    ix = ctx.cell_index('x')
    iy = ctx.cell_index('y')
    nx = ctx.size('x')
    ny = ctx.size('y')

    def stencil_var(key, freeze=False):
        'Returns: q, qxm, qxp, qym, qyp'
        if args.mg:
            uw = ctx.neural_net(key + 'w')()
            if freeze:
                pass
                #uw = tf.stop_gradient(uw)
            u = weights_to_field(uw, problem.nn, mod)
            st = [
                u,
                mod.roll(u, 1, 0),
                mod.roll(u, -1, 0),
                mod.roll(u, 1, 1),
                mod.roll(u, -1, 1),
            ]
        else:
            st = [
                ctx.field(key, freeze=freeze),
                ctx.field(key, -1, 0, freeze=freeze),
                ctx.field(key, 1, 0, freeze=freeze),
                ctx.field(key, 0, -1, freeze=freeze),
                ctx.field(key, 0, 1, freeze=freeze)
            ]
            if freeze:
                for i in range(len(st)):
                    pass
                    #st[i] = tf.stop_gradient(st[i])
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
        extrap = odil.util_op.extrap_quadh
        st[1] = mod.where(ix == 0, extrap(st[2], st[0], 0), st[1])
        st[2] = mod.where(ix == nx - 1, extrap(st[1], st[0], 0), st[2])
        st[3] = mod.where(iy == 0, extrap(st[4], st[0], 0), st[3])
        st[4] = mod.where(iy == ny - 1, extrap(st[3], st[0], 1), st[4])
        return st

    def apply_bc_v(st):
        'Quadratic extrapolation with boundary conditions.'
        extrap = odil.util_op.extrap_quadh
        st[1] = mod.where(ix == 0, extrap(st[2], st[0], 0), st[1])
        st[2] = mod.where(ix == nx - 1, extrap(st[1], st[0], 0), st[2])
        st[3] = mod.where(iy == 0, extrap(st[4], st[0], 0), st[3])
        st[4] = mod.where(iy == ny - 1, extrap(st[3], st[0], 0), st[4])
        return st

    def apply_bc_extrap(st):
        'Linear extrapolation from inner cells to halo cells.'
        extrap = odil.util_op.extrap_linear
        st[1] = mod.where(ix == 0, extrap(st[2], st[0]), st[1])
        st[2] = mod.where(ix == nx - 1, extrap(st[1], st[0]), st[2])
        st[3] = mod.where(iy == 0, extrap(st[4], st[0]), st[3])
        st[4] = mod.where(iy == ny - 1, extrap(st[3], st[0]), st[4])
        return st

    def stencil(q):
        'Returns: q, qxm, qxp, qym, qyp.'
        st = [None] * 5
        #q = tf.stop_gradient(q)
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
    apply_bc_u(u_st)
    apply_bc_u(uf_st)
    uf_st5 = stencil5(uf_st)
    uf = uf_st[0]

    v_st = stencil_var('v')
    vf_st = stencil_var('v', freeze=True)
    apply_bc_v(v_st)
    apply_bc_v(vf_st)
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
    return res


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Re',
                        type=float,
                        default=100,
                        help="Reynolds number")
    parser.add_argument('--N', type=int, default=65, help="Grid size")
    parser.add_argument('--mg', type=int, default=1, help="Use multigrid")
    parser.add_argument('--ref_path',
                        type=str,
                        default="cavity_N128_Re100.pickle",
                        help="Path to directory with reference solution")
    parser.add_argument('--rhiechow',
                        type=int,
                        default=1,
                        help="Use Rhie-Chow correction to remove "
                        "pressure oscillations")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(nlvl=1)

    parser.set_defaults(montage=0)
    parser.set_defaults(plotext='png')
    parser.set_defaults(frames=3)
    parser.set_defaults(checkpoint_every=0)
    parser.set_defaults(plot_every=100, report_every=10, history_every=10)
    parser.set_defaults(outdir='out_cavity2d')
    parser.set_defaults(history_full=50)

    #parser.set_defaults(optimizer='adam')
    parser.set_defaults(optimizer='lbfgsb')

    parser.set_defaults(every_factor=1)
    parser.set_defaults(lr=0.001)
    parser.set_defaults(linsolver='multigrid')
    parser.set_defaults(linsolver_maxiter=10)
    return parser.parse_args()


def state_to_field(key, problem, state):
    if key + 'w' in state.weights:
        uw = np.array(state.weights[key + 'w'][1])
        u = weights_to_field(uw, problem.nn, tf)
    else:
        u = state.fields[key]
    return np.array(u)


def plot(problem, state, epoch, frame):
    title0 = "u epoch={:05d}".format(epoch) if args.plot_title else None
    title1 = "v epoch={:05d}".format(epoch) if args.plot_title else None
    title2 = "p epoch={:05d}".format(epoch) if args.plot_title else None
    path0 = "u_{:05d}.{}".format(frame, args.plotext)
    path1 = "v_{:05d}.{}".format(frame, args.plotext)
    path2 = "p_{:05d}.{}".format(frame, args.plotext)
    printlog(path0, path1, path2)
    domain = problem.domain

    state_u = state_to_field('u', problem, state)
    state_v = state_to_field('v', problem, state)
    state_p = state_to_field('p', problem, state)

    exact_u = problem.exact_u
    exact_v = problem.exact_v
    exact_p = problem.exact_p

    umax = max(abs(np.max(exact_u)), abs(np.min(exact_u)))
    umax = max(0.01, umax)
    odil.util.plot(domain,
                   exact_u,
                   state_u,
                   path=path0,
                   title=title0,
                   cmap='RdBu_r',
                   nslices=5,
                   umin=-umax,
                   umax=umax)

    umax = max(abs(np.max(exact_v)), abs(np.min(exact_v)))
    umax = max(0.01, umax)
    odil.util.plot(domain,
                   exact_v,
                   state_v,
                   path=path1,
                   title=title1,
                   cmap='RdBu_r',
                   nslices=5,
                   umin=-umax,
                   umax=umax)

    state_p = state_p - state_p.mean()
    umin, umax = np.quantile(exact_p, 0.001), np.quantile(exact_p, 0.999)
    if umin == umax:
        umin, umax = np.quantile(p, 0.001), np.quantile(p, 0.999)
    if umax - umin < 2e-3:
        umin = -1e-3
        umax = 1e-3
    odil.util.plot(domain,
                   exact_p - np.mean(exact_p),
                   state_p,
                   path=path2,
                   title=title2,
                   nslices=5,
                   umin=umin,
                   umax=umax)
    if args.montage:
        cmd = [
            "montage", "-density", "300", "-geometry", "+0+0", path0, path1,
            path2, "tmp.jpg".format(epoch)
        ]
        cmd = [v for v in cmd if v is not None]
        printlog(' '.join(cmd))
        subprocess.run(cmd, check=True)
        cmd = ["mv", "tmp.jpg", "all_{:05d}.jpg".format(frame)]
        printlog(' '.join(cmd))
        subprocess.run(cmd, check=True)


def callback(packed, epoch, dhistory=None, opt=None, loss_grad=None):
    global g_time_start
    global frame, prevreport, history, problem, args

    report = (epoch % args.report_every == 0)
    calc = (epoch % args.report_every == 0 or epoch % args.history_every == 0
            or epoch < args.history_full
            or (epoch % args.plot_every == 0 and (epoch or args.frames)))

    if calc:
        state = problem.unpack_state(packed)
        if loss_grad is not None:
            loss = loss_grad(packed, epoch)[0].numpy()
        else:
            loss = 0
        state_u = state_to_field('u', problem, state)
        state_v = state_to_field('v', problem, state)
        state_p = state_to_field('p', problem, state)

    memusage = odil.util.get_memory_usage_kb()

    if report:
        printlog("\nepoch={:05d}".format(epoch))
        if opt.last_residual is not None:
            printlog('residual: ' + ', '.join('{:.5g}'.format(r)
                                              for r in opt.last_residual))
        printlog("walltime: {:.3f} s".format(time.time() - g_time_start))
        if epoch > prevreport.epoch:
            printlog("walltime/epoch: {:.3f} ms".format(
                (time.time() - prevreport.time) / (epoch - prevreport.epoch) *
                1000))
            prevreport.time = time.time()
            prevreport.epoch = epoch

    if epoch % args.plot_every == 0 and (epoch or args.frames):
        plot(problem, state, epoch, frame)
        frame += 1
    if args.checkpoint_every and epoch % args.checkpoint_every == 0:
        path = "state_{:06d}.pickle".format(epoch)
        printlog(path)
        odil.util_op.checkpoint_save(state, path)

    if ((epoch % args.history_every == 0 or epoch < args.history_full)
            and history is not None):
        exact_u = problem.exact_u
        exact_v = problem.exact_v
        exact_p = problem.exact_p
        history.append('epoch', epoch)
        history.append('frame', frame)
        history.append('loss', loss)
        if opt.last_residual is not None:
            for i, r in enumerate(np.array(opt.last_residual)):
                history.append('loss{:}'.format(i), r)
        history.append('walltime', time.time() - g_time_start)
        history.append('memory', memusage / 1024)
        history.append('u_err_linf', np.max(abs(exact_u - state_u)))
        history.append('u_err_l1', np.mean(abs(exact_u - state_u)))
        history.append('u_err_l2', np.mean((exact_u - state_u)**2)**0.5)
        history.append('v_err_linf', np.max(abs(exact_v - state_v)))
        history.append('v_err_l1', np.mean(abs(exact_v - state_v)))
        history.append('v_err_l2', np.mean((exact_v - state_v)**2)**0.5)
        history.append('p_err_linf', np.max(abs(exact_p - state_p)))
        history.append('p_err_l1', np.mean(abs(exact_p - state_p)))
        history.append('p_err_l2', np.mean((exact_p - state_p)**2)**0.5)
        if dhistory is not None:
            history.append_dict(dhistory)
        history.write()


def make_problem(args):
    nn = [((args.N - 1) >> level) + 1 for level in range(args.nlvl)]
    printlog('levels', nn)
    if args.mg:
        fieldnames = []
        neuralnets = {
            'uw': [0, sum([n**2 for n in nn])],
            'vw': [0, sum([n**2 for n in nn])],
            'pw': [0, sum([n**2 for n in nn])],
        }
    else:
        fieldnames = ['u', 'v', 'p']
        neuralnets = {}
    dtype = np.float64 if args.double else np.float32
    domain = odil.util_op.Domain(ndim=2,
                                 shape=[args.N, args.N],
                                 dtype=dtype,
                                 varnames=['x', 'y'],
                                 fieldnames=fieldnames,
                                 neuralnets=neuralnets)

    # Load reference solution.
    exact_uvp, path = load_reference(args.ref_path,
                                     domain,
                                     N=args.N,
                                     Re=args.Re)

    if exact_uvp is not None:
        printlog("Loading reference solution from '{}'".format(path))
    else:
        printlog("Can't load from '{}'. Setting to zeros".format(path))
        exact_uvp = [np.zeros(domain.shape) for _ in domain.fieldnames]
    exact_u, exact_v, exact_p = exact_uvp[:3]
    op = partial(operator_ns, args=args)
    problem = odil.util_op.Problem(op, domain)
    state = odil.util_op.State()
    problem.init_missing(state)
    problem.exact_u = exact_u
    problem.exact_v = exact_v
    problem.exact_p = exact_p
    problem.nn = nn
    return problem, state


def main():
    global frame, prevreport, history, problem, args

    args = parse_args()

    # Replace with absolute paths before changing current directory.
    if args.ref_path is not None:
        args.ref_path = os.path.abspath(args.ref_path)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # Change current directory to output directory.
    os.chdir(outdir)
    odil.util.set_log_file(open("train.log", 'w'))

    if args.history_every:
        history = odil.History(csvpath='train.csv', warmup=1)
    else:
        history = None
    prevreport = argparse.Namespace()
    prevreport.time = time.time()
    prevreport.epoch = 0
    prevreport.packed = None

    # Update arguments.
    args.mu = 1 / args.Re
    args.plot_every *= args.every_factor
    args.history_every *= args.every_factor
    args.report_every *= args.every_factor
    args.checkpoint_every *= args.every_factor
    if args.epochs is None:
        args.epochs = args.frames * args.plot_every

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    printlog(' '.join(sys.argv))

    problem, state = make_problem(args)
    frame = 0

    odil.util.optimize(args, args.optimizer, problem, state, callback)

    with open('done', 'w') as f:
        pass


if __name__ == "__main__":
    main()
