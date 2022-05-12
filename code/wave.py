#!/usr/bin/env python3

from collections import defaultdict
from util_cache import cache_to_file
from util import printlog, set_log_file
from util import TIME, TIMECLEAR
from util_op import extrap_linear, extrap_quadh, extrap_quad
import argparse
import json
import math
import numpy as np
import os
import subprocess
import sys
import tensorflow as tf
import time
import util_op

import linsolver
import matplotlib.pyplot as plt
import util

g_time_start = time.time()
g_time_callback = 0.


def get_exact(t, x):
    t = tf.Variable(t)
    x = tf.Variable(x)
    u = tf.zeros_like(x)
    with tf.GradientTape() as tape:
        ii = [1, 2, 3, 4, 5]
        for i in ii:
            k = i * np.pi
            u += tf.cos((x - t + 0.5) * k) + tf.cos((x + t - 0.5) * k)
        u /= 2 * len(ii)
    ut = tape.gradient(u, t).numpy()
    u = u.numpy()
    return u, ut


if 0:
    domain = util_op.Domain(ndim=2,
                            shape=(128, 128),
                            lower=(0, -1),
                            upper=(1, 1),
                            varnames=('t', 'x'))
    t, x = domain.cell_center_all()
    u, ut = get_exact(t, x)
    util.plot(domain,
              u,
              u,
              cmap='RdBu_r',
              path='exact_u.pdf',
              nslices=5,
              transpose=True)
    util.plot(domain,
              ut,
              ut,
              cmap='RdBu_r',
              path='exact_ut.pdf',
              nslices=5,
              transpose=True)
    exit()


def operator_fd(mod, ctx):
    global args, domain
    global init_u, init_ut, left_u, right_u
    dt = ctx.step('t')
    dx = ctx.step('x')
    x = ctx.cell_center('x')
    ones = ctx.field('ones')
    zeros = ctx.field('zeros')
    it = ctx.cell_index('t')
    ix = ctx.cell_index('x')
    nt = ctx.size('t')
    nx = ctx.size('x')

    def stencil_var(key):
        st = [
            ctx.field(key),
            ctx.field(key, -1, 0),
            ctx.field(key, -2, 0),
            ctx.field(key, -1, -1),
            ctx.field(key, -1, 1)
        ]
        return st

    left_utm = mod.roll(left_u, 1, axis=0)
    right_utm = mod.roll(right_u, 1, axis=0)

    def apply_bc_u(st):
        st[3] = mod.where(
            ix == 0,  #
            extrap_quadh(st[4], st[1], left_utm[:, None]),
            st[3])
        st[4] = mod.where(
            ix == nx - 1,  #
            extrap_quadh(st[3], st[1], right_utm[:, None]),
            st[4])
        return st

    u_st = stencil_var('u')
    apply_bc_u(u_st)
    u, utm, utmm, uxm, uxp = u_st

    u_t_tm = (u - utm) / dt
    u_t_tmm = (utm - utmm) / dt
    u_t_tmm = mod.where(it == 1, init_ut[None, :], u_t_tmm)

    u_tt = (u_t_tm - u_t_tmm) / dt
    u_xx = (uxm - 2 * utm + uxp) / (dx**2)

    fu = u_tt - u_xx

    u0 = init_u + 0.5 * dt * init_ut
    fu = mod.where(it == 0, u - u0[None, :], fu)

    res = [fu]

    return res


def operator_pinn(mod, ctx):
    global t_in, x_in
    global t_bound, x_bound, u_bound
    global t_init, x_init, u_init, ut_init
    u = ctx.neural_net('u', t_in, x_in)
    utt = u(2, 0)
    uxx = u(0, 2)
    fu = utt - uxx

    ub = ctx.neural_net('u', t_bound, x_bound)
    fb = ub(0, 0) - u_bound

    ui = ctx.neural_net('u', t_init, x_init)
    fiu = ui(0, 0) - u_init
    fiut = ui(1, 0) - ut_init

    return fu, fb, fiu, fiut


def get_uut(domain, uu):
    global init_u
    dt = domain.step_by_dim(0)
    u = uu
    utm = np.roll(u, 1, axis=0)
    utp = np.roll(u, -1, axis=0)
    utm[0, :] = extrap_quadh(utp[0, :], u[0, :], init_u)
    utp[-1, :] = extrap_quad(u[-3, :], u[-2, :], u[-1, :])
    uut = (utp - utm) / (2 * dt)
    return uut


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Nt', type=int, default=128, help="Grid size in t")
    parser.add_argument('--Nx', type=int, default=128, help="Grid size in x")
    parser.add_argument('--Nci',
                        type=int,
                        default=4096,
                        help="Number of collocation points inside domain")
    parser.add_argument('--Ncb',
                        type=int,
                        default=128,
                        help="Number of collocation points on each boundary")
    parser.add_argument('--arch',
                        type=int,
                        nargs="*",
                        default=[10, 10],
                        help="Network architecture, "
                        "number of neurons in hidden layers")
    parser.add_argument('--solver',
                        type=str,
                        choices=('pinn', 'fd'),
                        default='fd',
                        help="Grid size in x")
    util.add_arguments(parser)
    linsolver.add_arguments(parser)
    return parser.parse_args()


@tf.function()
def u_deriv(weights, tt, xx, nt, nx):
    net_u = util_op.NeuralNet(weights, tt, xx)
    return net_u(nt, nx)


def plot(state, epoch, frame, exact_uu, exact_uut):
    global domain

    title0 = "u epoch={:05d}".format(epoch) if args.plot_title else None
    title1 = "u_t epoch={:05d}".format(epoch) if args.plot_title else None
    path0 = "u_{:05d}.pdf".format(frame)
    path1 = "v_{:05d}.pdf".format(frame)
    printlog(path0, path1)

    if args.solver == 'fd':
        uu, = [np.array(state.fields[name]) for name in domain.fieldnames]
        uut = get_uut(domain, uu)
    elif args.solver == 'pinn':
        tt, xx = domain.cell_center_all()
        eval_u = lambda nt, nx: u_deriv(  #
            state.weights['u'], tt, xx, nt, nx).numpy()
        uu = eval_u(0, 0)
        uut = eval_u(1, 0)

    umax = max(abs(np.max(exact_uu)), abs(np.min(exact_uu)))
    util.plot(domain,
              exact_uu,
              uu,
              path=path0,
              title=title0,
              cmap='RdBu_r',
              nslices=5,
              transpose=True,
              umin=-umax,
              umax=umax)

    x_imp = None
    y_imp = None
    marker_color = None
    if args.solver == 'pinn':
        x_imp = [x_in, x_init, x_bound]
        y_imp = [t_in, t_init, t_bound]
        marker_color = ['r', 'g', 'b']

    umax = max(abs(np.max(exact_uut)), abs(np.min(exact_uut)))
    util.plot(domain,
              exact_uut,
              uut,
              path=path1,
              title=title1,
              cmap='RdBu_r',
              nslices=5,
              x_imp=x_imp,
              y_imp=y_imp,
              marker_color=marker_color,
              transpose=True,
              umin=-umax,
              umax=umax)

    if args.montage:
        cmd = [
            "montage", "-density", "300", "-geometry", "+0+0", path0, path1,
            "tmp.jpg".format(epoch)
        ]
        cmd = [v for v in cmd if v is not None]
        printlog(' '.join(cmd))
        subprocess.run(cmd, check=True)
        cmd = ["mv", "tmp.jpg", "all_{:05d}.jpg".format(frame)]
        printlog(' '.join(cmd))
        subprocess.run(cmd, check=True)


def callback(packed, epoch, dhistory=None, opt=None, loss_grad=None):
    tstart = time.time()

    def callback_update_time():
        nonlocal tstart
        global g_time_callback
        t = time.time()
        g_time_callback += t - tstart
        tstart = t

    global frame, csv, csv_empty, packed_prev, domain
    global history, nhistory
    global g_time_callback, g_time_start
    global exact_uu, exact_uut

    report = (epoch % args.report_every == 0)
    calc = (epoch % args.report_every == 0 or epoch % args.history_every == 0
            or epoch < args.history_full
            or (epoch % args.plot_every == 0 and (epoch or args.frames)))
    loss = 0

    if packed_prev is None:
        packed_prev = packed

    if calc:
        state = problem.unpack_state(packed)
        state_prev = problem.unpack_state(packed_prev)
        if loss_grad is not None:
            loss = loss_grad(packed, epoch)[0].numpy()

    packed_prev = np.copy(packed)

    if report:
        printlog("epoch={:05d}".format(epoch))
    if epoch % args.plot_every == 0 and (epoch or args.frames):
        plot(state, epoch, frame, exact_uu, exact_uut)
        frame += 1

    if report:
        printlog("T last: " + ', '.join([
            '{}={:7.3f}'.format(k, t)
            for k, t in problem.timer_last.counters.items()
        ]))
        printlog("T  all: " + ', '.join([
            '{}={:7.3f}'.format(k, t)
            for k, t in problem.timer_total.counters.items()
        ]))

    if calc:
        if args.solver == 'fd':
            uu, = [np.array(state.fields[name]) for name in domain.fieldnames]
            uu_prev, = [
                np.array(state_prev.fields[name]) for name in domain.fieldnames
            ]
        elif args.solver == 'pinn':
            tt, xx = domain.cell_center_all()
            uu = util_op.eval_neural_net(state.weights['u'], tt, xx).numpy()
            uu_prev = util_op.eval_neural_net(state_prev.weights['u'], tt,
                                              xx).numpy()
        du = uu - uu_prev

    if report:
        printlog("u={:.05g} du={:.05g}".format(np.max(np.abs(uu)),
                                               np.max(np.abs(du))))

    callback_update_time()

    if report:
        printlog()

    if (epoch % args.history_every == 0
            or epoch < args.history_full) and csv is not None:
        assert calc
        history['epoch'].append(epoch)
        history['du_linf'].append(np.max(abs(du)))
        history['du_l1'].append(np.mean(abs(du)))
        history['du_l2'].append(np.mean((du)**2)**0.5)
        history['t_linsolver'].append(
            problem.timer_last.counters.get('linsolver', 0))
        history['t_grad'].append(
            problem.timer_last.counters.get('eval_grad', 0))
        history['t_sparse_fields'].append(
            problem.timer_last.counters.get('sparse_fields', 0))
        history['t_sparse_weights'].append(
            problem.timer_last.counters.get('sparse_weights', 0))

        history['ref_du_linf'].append(np.max(abs(exact_uu - uu)))
        history['ref_du_l1'].append(np.mean(abs(exact_uu - uu)))
        history['ref_du_l2'].append(np.mean((exact_uu - uu)**2)**0.5)

        history['loss'].append(loss)

        callback_update_time()
        history['tt_linsolver'].append(
            problem.timer_total.counters.get('linsolver', 0))
        history['tt_opt'].append(time.time() - g_time_start - g_time_callback)
        history['tt_callback'].append(g_time_callback)
        if opt:
            history['evals'].append(opt.evals)
        else:
            history['evals'].append(0)

        if dhistory is not None:
            for k, v in dhistory.items():
                history[k].append(v)
        nhistory += 1

        keys = list(history)
        if csv_empty:
            csv.write(','.join(keys) + '\n')
            csv_empty = False
        for k in history:
            kref = 'epoch'
            assert len(history[k]) == len(history[kref]), \
                "Wrong history size: {:} of '{}' and {:} of '{}'".format(
                    len(history[k]), k, len(history[kref]), kref)
        row = [history[key][-1] for key in keys]
        line = ','.join(map(str, row))
        csv.write(line + '\n')
        csv.flush()
        callback_update_time()


def main():
    global csv, csv_empty, packed_prev, args, problem, domain, frame
    global history, nhistory
    global exact_uu, exact_uut
    global init_u, init_ut, left_u, right_u
    global t_in, x_in
    global t_init, x_init, u_init, ut_init
    global t_bound, x_bound, u_bound

    args = parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # Change current directory to output directory.
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    set_log_file(open("train.log", 'w'))

    csv = None
    csv_empty = True
    if args.history_every:
        csv = open('train.csv', 'w')
    nhistory = 0
    history = defaultdict(lambda: [0 for _ in range(nhistory)])
    packed_prev = None

    # Update arguments.
    args.plot_every *= args.every_factor
    args.history_every *= args.every_factor
    args.report_every *= args.every_factor
    if args.epochs is None:
        args.epochs = args.frames * args.plot_every

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    if args.solver == 'fd':
        fieldnames = ['u']
        neuralnets = {}
        operator = operator_fd
    elif args.solver == 'pinn':
        fieldnames = []
        neuralnets = {
            'u': [2] + args.arch + [1],
        }
        operator = operator_pinn
    else:
        assert False

    domain = util_op.Domain(ndim=2,
                            shape=(args.Nt, args.Nx),
                            lower=(0, -1),
                            upper=(1, 1),
                            varnames=('t', 'x'),
                            fieldnames=fieldnames,
                            neuralnets=neuralnets)

    if args.solver == 'pinn':
        t_in, x_in = domain.random_inner(args.Nci)
        t_bound, x_bound = domain.random_boundary(1, 0, args.Ncb)
        t_bound2, x_bound2 = domain.random_boundary(1, 1, args.Ncb)
        t_bound = np.hstack((t_bound, t_bound2))
        x_bound = np.hstack((x_bound, x_bound2))
        t_init, x_init = domain.random_boundary(0, 0, args.Ncb)
        u_init, ut_init = get_exact(t_init, x_init)
        u_bound, _ = get_exact(t_bound, x_bound)
        printlog('Number of collocation points:')
        printlog('inside: {:}'.format(len(t_in)))
        printlog('init: {:}'.format(len(t_init)))
        printlog('bound: {:}'.format(len(t_bound)))

    printlog(' '.join(sys.argv))

    # Evaluate exact solution, boundary and initial conditions.
    t, x = domain.cell_center_all()
    t1 = domain.cell_center_1d(0)
    x1 = domain.cell_center_1d(1)
    exact_uu, exact_uut = get_exact(t, x)
    left_u, _ = get_exact(t1, t1 * 0 + domain.lower[1])
    right_u, _ = get_exact(t1, t1 * 0 + domain.upper[1])
    init_u, init_ut = get_exact(x1 * 0 + domain.lower[0], x1)

    problem = util_op.Problem(operator, domain)

    state = util_op.State()
    frame = 0

    problem.init_missing(state)

    args.ntrainable = len(problem.pack_state(state))
    with open('args.json', 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    if args.optimizer == 'newton':
        util.optimize_newton(args, problem, state, callback)
    else:
        util.optimize_opt(args, args.optimizer, problem, state, callback)

    with open('done', 'w') as f:
        pass


if __name__ == "__main__":
    main()
