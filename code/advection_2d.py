#!/usr/bin/env python3

import argparse
import json
import math
import numpy as np
import scipy.optimize
import sys
import util_op
import tensorflow as tf
from util_cache import cache_to_file
from scipy.interpolate import interp1d, RectBivariateSpline
import subprocess
from util import TIME, TIMECLEAR
import os
from util import printlog, set_log_file
from collections import defaultdict
from util_op import extrap_linear

import util
import linsolver
import matplotlib.pyplot as plt


def get_anneal_factor(epoch):
    return 0.5**(epoch / args.anneal_half) if args.anneal_half else 1


def get_exact_preset(name, domain, args):
    tt, xx, yy = domain.cell_center_all()
    zeros = xx * 0
    if name == 'uniform':
        # Uniform velocity.
        u0 = 1.
        v0 = 0.5
        uu = u0 + zeros
        vv = v0 + zeros
        x = 2 * np.pi * xx
        y = 2 * np.pi * yy
        pp = np.sin(x + y) * np.sin(y)
        return uu, vv, pp
    if name == 'sin_uniform':
        # Uniform velocity and sine wave scalar field.
        u0 = 0.5  # Velocity.
        v0 = 0.
        kx = 2.  # Wave number.
        ky = 1.
        uu = u0 + zeros
        vv = v0 + zeros
        x = 2 * np.pi * (xx - tt * u0)
        y = 2 * np.pi * (yy - tt * v0)
        pp = np.cos(x * kx + y * ky)
        return uu, vv, pp
    if name == 'sin_radial':
        # Radial velocity field and radial sine wave scalar field.
        k = 2.
        dx = xx - 0.5
        dy = yy - 0.5
        ts = 0.5
        rr = (dx**2 + dy**2)**0.5
        #pp = np.cos(2 * np.pi * k * rr / (tt + ts))
        pp = np.cos(2 * np.pi * k * rr / ts)
        q = (xx * (1 - xx) * yy * (1 - yy) * 16)**0.5
        uu = dx / (tt + ts) * q
        vv = dy / (tt + ts) * q
        return uu, vv, pp
    if name == 'blob':
        # Single blob advected by uniform velocity field.
        u0 = 0.25
        v0 = 0.125
        r0 = 0.23
        uu = u0 + zeros
        vv = v0 + zeros
        x = xx - u0 * tt
        y = yy - v0 * tt
        k = tt / args.tmax + 1
        x *= k
        y /= k
        pp = np.maximum(0, 1 - ((x - 0.25)**2 + (y - 0.25)**2) / r0**2)
        return uu, vv, pp
    if name == 'uniform_accel':
        # Velocity with uniform acceleration in time.
        u0 = 1.
        v0 = 0.5
        t = tt / tt.max()
        uu = u0 * t + zeros
        vv = v0 * t + zeros
        x = 2 * np.pi * xx
        y = 2 * np.pi * yy
        pp = np.sin(x + y) * np.sin(y)
        return uu, vv, pp
    if name == 'vortex':
        # Single vortex.
        x = 2 * np.pi * xx
        y = 2 * np.pi * yy
        pp = np.sin(x + y) * np.sin(y)
        uu = np.sin(x) * np.cos(y)
        vv = -np.cos(x) * np.sin(y)
        return uu, vv, pp
    return None


def operator(mod, ctx):
    global args
    dt = ctx.step('t')
    dx = ctx.step('x')
    dy = ctx.step('y')
    ones = ctx.field('ones')
    zeros = ctx.field('zeros')
    it = ctx.cell_index('t')
    ix = ctx.cell_index('x')
    iy = ctx.cell_index('y')
    nt = ctx.size('t')
    nx = ctx.size('x')
    ny = ctx.size('y')
    epoch = ctx.epoch

    def stencil_var(key, freeze=False, shift_t=0):
        'Returns: q, qxm, qxp, qym, qyp'
        st = [
            ctx.field(key, shift_t, 0, 0, freeze=freeze),
            ctx.field(key, shift_t, -1, 0, freeze=freeze),
            ctx.field(key, shift_t, 1, 0, freeze=freeze),
            ctx.field(key, shift_t, 0, -1, freeze=freeze),
            ctx.field(key, shift_t, 0, 1, freeze=freeze)
        ]
        return st

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

    def apply_bc_extrap(st):
        'Linear extrapolation from inner cells to halo cells.'
        st[1] = mod.where(ix == 0, extrap_linear(st[2], st[0]), st[1])
        st[2] = mod.where(ix == nx - 1, extrap_linear(st[1], st[0]), st[2])
        st[3] = mod.where(iy == 0, extrap_linear(st[4], st[0]), st[3])
        st[4] = mod.where(iy == ny - 1, extrap_linear(st[3], st[0]), st[4])
        return st

    def apply_bc_zero(st):
        'Linear extrapolation from inner cells to halo cells.'
        st[1] = mod.where(ix == 0, zeros, st[1])
        st[2] = mod.where(ix == nx - 1, zeros, st[2])
        st[3] = mod.where(iy == 0, zeros, st[3])
        st[4] = mod.where(iy == ny - 1, zeros, st[4])
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

    def central(st):
        q, qxm, qxp, qym, qyp = st
        q_x = (qxp - qxm) / (2 * dx)
        q_y = (qyp - qym) / (2 * dy)
        return q_x, q_y

    def laplace(st):
        q, qxm, qxp, qym, qyp = st
        q_xx = (qxp - 2 * q + qxm) / dx**2
        q_yy = (qyp - 2 * q + qym) / dy**2
        q_lap = q_xx + q_yy
        return q_lap

    u_st = stencil_var('u')
    uf_st = stencil_var('u', freeze=True)
    if not args.periodic:
        apply_bc_zero(u_st)
        apply_bc_zero(uf_st)
    u = u_st[0]
    uf = uf_st[0]

    v_st = stencil_var('v')
    vf_st = stencil_var('v', freeze=True)
    if not args.periodic:
        apply_bc_zero(v_st)
        apply_bc_zero(vf_st)
    v = v_st[0]
    vf = vf_st[0]

    p_st = stencil_var('p')
    ptm_st = stencil_var('p', shift_t=-1)
    if not args.periodic:
        apply_bc_zero(p_st)
        apply_bc_zero(ptm_st)
    exact_pp_st = stencil(exact_pp)
    for i in range(len(ptm_st)):
        ptm_st[i] = mod.where(it == 0, exact_pp_st[i], ptm_st[i])
    p = p_st[0]
    ptm = ptm_st[0]

    if args.central:
        p_x, p_y = central(p_st)
    else:
        p_x, p_y = upwind_low(p_st, uf, vf)
    if args.implicit:
        if args.central:
            ptm_x, ptm_y = central(ptm_st)
        else:
            ptm_x, ptm_y = upwind_low(ptm_st, uf, vf)
        p_x = (p_x + ptm_x) * 0.5
        p_y = (p_y + ptm_y) * 0.5
    p_t = (p - ptm) / dt

    if args.problem == 'direct':
        fp = p_t + uf * p_x + vf * p_y
        fu = u - exact_uu
        fv = v - exact_vv
        return fp, fu, fv
    elif args.problem == 'inverse':
        fp = p_t + u * p_x + v * p_y
        anneal_factor = get_anneal_factor(ctx.epoch)
        fe = (p - exact_pp) * args.alpha * anneal_factor
        if args.onlyfinal:
            fe = mod.where((it == 0) | (it == nt - 1), fe, zeros)
        res = [fe, fp]

        if args.kreg:
            fsu = laplace(u_st) * args.kreg
            fsv = laplace(v_st) * args.kreg
            res += [fsu, fsv]

        if args.ktreg:
            ftregu = (u - ctx.field('u', -1, 0, 0)) / dt * args.ktreg
            ftregv = (v - ctx.field('v', -1, 0, 0)) / dt * args.ktreg
            ftregu = mod.where(it == 0, zeros, ftregu)
            ftregv = mod.where(it == 0, zeros, ftregv)
            res += [ftregu, ftregv]

        if args.divfree:
            u_xc, _ = central(u_st)
            _, v_yc = central(v_st)
            fdiv = (u_xc + v_yc) * args.divfree
            res.append(fdiv)

        return res
    else:
        raise ValueError("Unknown problem=" + args.problem)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',
                        type=float,
                        default=1,
                        help="Weight to impose known solution")
    parser.add_argument('--Nx', type=int, default=32, help="Grid size in x")
    parser.add_argument('--Ny', type=int, default=32, help="Grid size in y")
    parser.add_argument('--Nt', type=int, default=32, help="Grid size in t")
    parser.add_argument('--onlyfinal',
                        type=int,
                        default=0,
                        help="Only impose initial and final solution")
    parser.add_argument('--quiverref',
                        type=int,
                        default=1,
                        help="Quiver plot of reference solution")
    parser.add_argument('--implicit',
                        type=int,
                        default=0,
                        help="Implicit discretization in time")
    parser.add_argument('--ref_path',
                        type=str,
                        default="uniform",
                        help="Path to directory with reference solution")
    parser.add_argument('--load_fields',
                        type=str,
                        help="Path to checkpoint with initial fields")
    parser.add_argument('--periodic',
                        type=int,
                        default=1,
                        help="Periodic conditions")
    parser.add_argument('--central',
                        type=int,
                        default=0,
                        help="Central differences in space")
    parser.add_argument('--divfree',
                        type=float,
                        default=0,
                        help="Weight of divergence-free condition")
    parser.add_argument('--problem',
                        type=str,
                        default='direct',
                        choices=('direct', 'inverse'),
                        help="Problem to solve")
    parser.add_argument('--anneal_half',
                        type=float,
                        default=0,
                        help="Number of epochs before halving "
                        "of regularization factors")
    parser.add_argument('--kreg',
                        type=float,
                        default=1e-4,
                        help="Laplacian regularization factor")
    parser.add_argument('--ktreg',
                        type=float,
                        default=0.0,
                        help="Time regularization factor")
    parser.add_argument('--tmax',
                        type=float,
                        default=1,
                        help="Simulation time")
    util.add_arguments(parser)
    linsolver.add_arguments(parser)
    return parser.parse_args()


args = parse_args()

outdir = args.outdir
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, sort_keys=True, indent=4)

if args.seed is not None:
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

args.epochs = args.frames * args.plot_every

domain = util_op.Domain(ndim=3,
                        shape=(args.Nt, args.Nx, args.Ny),
                        upper=(args.tmax, 1., 1.),
                        dtype=np.float64,
                        varnames=('t', 'x', 'y'),
                        fieldnames=('u', 'v', 'p'))

path = args.ref_path
exact_uvp = None
if os.path.isfile(path):
    printlog("Loading reference solution from '{}'".format(path))
    state = util_op.State()
    util_op.checkpoint_load(state, path, fields_to_load=domain.fieldnames)
    exact_uvp = state.fields['u'], state.fields['v'], state.fields['p']
    exact_loaded = True

if exact_uvp is None:
    exact_uvp = get_exact_preset(path, domain, args)
    if exact_uvp is not None:
        printlog("Loading preset solution '{}'".format(path))
        exact_loaded = True

if exact_uvp is None:
    raise ValueError("Can't load reference solution from '{}'".format(path))

exact_uu, exact_vv, exact_pp = exact_uvp

Nf = len(domain.fieldnames)
N = np.prod(domain.shape)

shape = domain.shape
problem = util_op.Problem(operator, domain)

state = util_op.State()

if args.load_fields is not None:
    printlog("Loading initial fields from '{}'".format(args.load_fields))
    util_op.checkpoint_load(state,
                            args.load_fields,
                            fields_to_load=domain.fieldnames)

problem.init_missing(state)
frame = 0


def plot(state, epoch, frame):
    title0 = "u epoch={:05d}".format(epoch)
    title1 = "v epoch={:05d}".format(epoch)
    title2 = "p epoch={:05d}".format(epoch)
    title3 = "k epoch={:05d}".format(epoch)
    path0 = "u_{:05d}.png".format(frame)
    path1 = "v_{:05d}.png".format(frame)
    path2 = "p_{:05d}.png".format(frame)
    printlog(path0, path1, path2)
    slices_it = np.linspace(0, domain.shape[0] - 1, 5, dtype=int)
    it = args.Nt // 2
    path0s = "u_series_{:05d}.png".format(frame)
    path1s = "v_series_{:05d}.png".format(frame)
    path2s = "p_series_{:05d}.png".format(frame)
    path3s = "quv_series_{:05d}.pdf".format(frame)
    uu, vv, pp = [np.array(state.fields[name]) for name in domain.fieldnames]

    # Plot velocity u.
    if exact_loaded:
        umax = max(abs(np.max(exact_uu)), abs(np.min(exact_uu)),
                   abs(np.max(exact_vv)), abs(np.min(exact_vv)))
    else:
        umax = max(abs(np.max(uu)), abs(np.min(uu)), abs(np.max(vv)),
                   abs(np.min(vv)))
    umax = max(0.01, umax)
    util.plot(domain,
              exact_uu[it],
              uu[it],
              path=path0,
              title=title0,
              transpose=True,
              cmap='RdBu_r',
              nslices=5,
              umin=-umax,
              umax=umax)

    util.plot_2d(domain,
                 exact_uu,
                 uu,
                 slices_it,
                 path0s,
                 umin=-umax,
                 umax=umax,
                 cmap='RdBu_r')

    # Plot velocity v.
    util.plot(domain,
              exact_vv[it],
              vv[it],
              path=path1,
              title=title1,
              transpose=True,
              cmap='RdBu_r',
              nslices=5,
              umin=-umax,
              umax=umax)
    util.plot_2d(domain,
                 exact_vv,
                 vv,
                 slices_it,
                 path1s,
                 umin=-umax,
                 umax=umax,
                 cmap='RdBu_r')

    # Plot tracer p.
    if exact_loaded:
        umin = np.min(exact_pp)
        umax = np.max(exact_pp)
    else:
        umin = np.min(pp)
        umax = np.max(pp)
    if umax - umin < 0.002:
        umax += 0.001
        umin += 0.001
    util.plot(domain,
              exact_pp[it],
              pp[it],
              path=path2,
              title=title2,
              transpose=True,
              nslices=5,
              umin=umin,
              umax=umax)

    util.plot_2d(domain, exact_pp, pp, slices_it, path2s, umin=umin, umax=umax)

    def callback(i, j, ax, fig):
        plt.setp(ax.spines.values(), linewidth=0.25)
        ax.yaxis.label.set_size(7)
        _, xx, yy = domain.cell_center_all()
        skip = domain.shape[1] // 8
        off = skip // 2 - 1
        x = xx[0, off::skip, off::skip].flatten()
        y = yy[0, off::skip, off::skip].flatten()
        if args.onlyfinal and i == 0 and j > 0 and j < len(slices_it) - 1:
            ax.remove()
            return
        if i == 0 and not args.quiverref:
            return
        vx = exact_uu if i == 0 else uu
        vy = exact_vv if i == 0 else vv
        vx = vx[slices_it[j], off::skip, off::skip].flatten()
        vy = vy[slices_it[j], off::skip, off::skip].flatten()
        ax.quiver(x, y, vx, vy, scale=8., color='k')

    util.plot_2d(domain,
                 exact_pp,
                 pp,
                 slices_it,
                 path3s,
                 xlabel="",
                 figsizey=1.6,
                 cmap="Blues",
                 callback=callback,
                 umin=umin,
                 umax=umax)

    if args.montage:
        cmd = [
            "convert", '(', path0, path1, path2, '+append', ')', '(', path0s,
            path1s, path2s, '+append', ')', "-geometry", "1800x", "-append",
            "tmp.jpg".format(epoch)
        ]
        printlog(' '.join(cmd))
        subprocess.run(cmd, check=True)
        cmd = ["mv", "tmp.jpg", "all_{:05d}.jpg".format(frame)]
        printlog(' '.join(cmd))
        subprocess.run(cmd, check=True)


# Replace with absolute paths before changing current directory.
if args.ref_path is not None:
    args.ref_path = os.path.abspath(args.ref_path)

# Change current directory to output directory.
os.makedirs(outdir, exist_ok=True)
os.chdir(outdir)
set_log_file(open("train.log", 'w'))

csv = None
csv_empty = True
if args.history_every:
    csv = open('train.csv', 'w')
history = defaultdict(lambda: [])


def optimize_newton():
    global frame, csv, csv_empty, history
    for epoch in range(args.epochs + 1):

        def printrep(m=''):
            if epoch % args.report_every == 0:
                printlog(m)

        printrep("epoch={:05d}".format(epoch))
        if epoch % args.plot_every == 0:
            plot(state, epoch, frame)
            frame += 1
        if epoch % args.checkpoint_every == 0:
            path = "state_{:06d}.pickle".format(epoch)
            printlog(path)
            util_op.checkpoint_save(state, path)

        if epoch == args.epochs:
            break

        const, m = problem.linearize(state, epoch=epoch)
        packed = problem.pack_state(state)
        const = np.array(const).flatten()

        timer = util.Timer()
        timer.push("solve")
        dpacked = linsolver.solve(m, -const, args, history, args.linsolver)
        timer.pop()
        problem.timer_total.append(timer)
        problem.timer_last.append(timer)

        printrep("T last: " + ', '.join([
            '{}={:7.3f}'.format(k, t)
            for k, t in problem.timer_last.counters.items()
        ]))
        printrep("T  all: " + ', '.join([
            '{}={:7.3f}'.format(k, t)
            for k, t in problem.timer_total.counters.items()
        ]))

        packed += dpacked
        dstate = problem.unpack_state(dpacked)
        du, dv, dp = [
            np.array(dstate.fields[name]) for name in domain.fieldnames
        ]
        state.fields['u'].assign_add(du)
        state.fields['v'].assign_add(dv)
        state.fields['p'].assign_add(dp)
        uu, vv, pp = [
            np.array(state.fields[name]) for name in domain.fieldnames
        ]
        printrep("u={:.05g} du={:.05g}".format(np.mean(uu),
                                               np.max(np.abs(du))))
        printrep("v={:.05g} dv={:.05g}".format(np.mean(vv),
                                               np.max(np.abs(dv))))
        printrep("p={:.05g} dp={:.05g}".format(np.mean(pp),
                                               np.max(np.abs(dp))))

        history['epoch'].append(epoch)
        history['du'].append(np.max(np.abs(du)))
        history['dv'].append(np.max(np.abs(dv)))
        history['dp'].append(np.max(np.abs(dp)))
        history['t_solve'].append(problem.timer_last.counters['solve'])
        history['t_grad'].append(problem.timer_last.counters['eval_grad'])
        history['t_sparse_fields'].append(
            problem.timer_last.counters['sparse_fields'])

        printrep()

        if epoch % args.history_every == 0 and csv is not None:
            keys = list(history)
            if csv_empty:
                csv.write(','.join(keys) + '\n')
                csv_empty = False
            row = [history[key][-1] for key in keys]
            line = ','.join(map(str, row))
            csv.write(line + '\n')
            csv.flush()


def optimize_opt(factr=10, pgtol=1e-16, m=50, maxls=50):
    global frame, csv, csv_empty, history, state
    epoch = 0

    def printrep(m=''):
        if epoch % args.report_every == 0:
            printlog(m)

    def func(x):
        s = problem.unpack_state(x)
        loss, grads, _ = problem.eval_loss_grad(s, epoch)
        g = problem.pack_fields(grads)
        return loss, g

    x_prev = None

    def callback(x):
        global csv, csv_empty, history
        nonlocal x_prev, epoch
        global frame
        printrep("epoch={:05d}".format(epoch))
        epoch += 1
        loss, _ = func(x)
        s = problem.unpack_state(x)
        if epoch % args.plot_every == 0:
            plot(s, epoch, frame)
            frame += 1
        printrep("T last: " + ', '.join([
            '{}={:7.3f}'.format(k, t)
            for k, t in problem.timer_last.counters.items()
        ]))
        printrep("T  all: " + ', '.join([
            '{}={:7.3f}'.format(k, t)
            for k, t in problem.timer_total.counters.items()
        ]))

        dstate = problem.unpack_state(x - x_prev)
        du, dv, dp = [
            np.array(dstate.fields[name]) for name in domain.fieldnames
        ]
        state.fields['u'].assign_add(du)
        state.fields['v'].assign_add(dv)
        state.fields['p'].assign_add(dp)
        uu, vv, pp = [
            np.array(state.fields[name]) for name in domain.fieldnames
        ]
        printrep("u={:.05g} du={:.05g}".format(np.mean(uu),
                                               np.max(np.abs(du))))
        printrep("v={:.05g} dv={:.05g}".format(np.mean(vv),
                                               np.max(np.abs(dv))))
        printrep("p={:.05g} dp={:.05g}".format(np.mean(pp),
                                               np.max(np.abs(dp))))

        history['epoch'].append(epoch)
        history['du'].append(np.max(np.abs(du)))
        history['dv'].append(np.max(np.abs(dv)))
        history['dp'].append(np.max(np.abs(dp)))
        history['anneal'].append(get_anneal_factor(epoch))
        history['t_grad'].append(problem.timer_last.counters['eval_loss_grad'])

        printrep()

        if epoch % args.history_every == 0 and csv is not None:
            keys = list(history)
            if csv_empty:
                csv.write(','.join(keys) + '\n')
                csv_empty = False
            row = [history[key][-1] for key in keys]
            line = ','.join(map(str, row))
            csv.write(line + '\n')
            csv.flush()

    x0 = problem.pack_state(state)
    x_prev = x0

    x, loss, info = scipy.optimize.fmin_l_bfgs_b(func=func,
                                                 x0=x0,
                                                 factr=factr,
                                                 pgtol=pgtol,
                                                 m=m,
                                                 maxls=maxls,
                                                 maxiter=args.epochs,
                                                 callback=callback)
    state = problem.unpack_state(x)
    printrep(info)

if args.optimizer == 'newton':
    optimize_newton()
elif args.optimizer == 'lbfgsb':
    optimize_opt()
else:
    assert False, "unknown optimizer=" + args.optimizer
