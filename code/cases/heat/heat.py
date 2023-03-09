#!/usr/bin/env python3

from collections import defaultdict
import argparse
import json
import math
import numpy as np
import os
import subprocess
import sys
import time
import pickle

import odil
from odil import tf
from odil import MultigridDecomp
import matplotlib.pyplot as plt

g_time_start = time.time()
g_time_callback = 0.

printlog = odil.util.printlog


def get_init_u(t, x):
    # Step function.
    #return np.where(abs(x - 0.5) < 0.15, 1, 0)

    # Gaussian.
    def f(z):
        return np.exp(-(z - 0.5)**2 * 50)

    return f(x) - f(-0.5)


def get_ref_k(u, mod=np):
    #return 0.02 * mod.ones_like(u)
    # ksigmoid
    #return 0.02 * 1 / (1 + mod.exp((0.5 - u) * 10))
    # kgauss
    return 0.02 * (mod.exp(-(u - 0.5)**2 * 20))


def get_anneal_factor(epoch, anneal_half):
    return 0.5**(epoch / anneal_half) if anneal_half else 1


def knet_to_k(knet):
    return knet**2


def operator_odil(mod, ctx):
    global args, domain
    global init_u
    dt = ctx.step('t')
    dx = ctx.step('x')
    x = ctx.cell_center('x')
    ones = ctx.field('ones')
    zeros = ctx.field('zeros')
    it = ctx.cell_index('t')
    ix = ctx.cell_index('x')
    nt = ctx.size('t')
    nx = ctx.size('x')

    def stencil_var(key, freeze=False):
        if args.mg:
            qw = ctx.neural_net(key)()
            q = MultigridDecomp.weights_to_cell_field(qw, problem.nnw, mod)
            st = [
                q,
                mod.roll(q, 1, 0),
                mod.roll(q, 1, 1),
                mod.roll(q, -1, 1),
            ]
        else:
            st = [
                ctx.field(key, 0, 0, freeze=freeze),
                ctx.field(key, -1, 0, freeze=freeze),
                ctx.field(key, 0, -1, freeze=freeze),
                ctx.field(key, 0, 1, freeze=freeze),
            ]
        return st

    def apply_bc_u(st):
        extrap = odil.util_op.extrap_quadh
        # Zero Dirichlet conditions.
        st[1] = mod.where(it == 0, extrap(st[1], st[0], init_u), st[1])
        st[2] = mod.where(ix == 0, extrap(st[2], st[0], 0), st[2])
        st[3] = mod.where(ix == nx - 1, extrap(st[2], st[0], 0), st[3])
        return st

    u_st = stencil_var('u')
    apply_bc_u(u_st)
    u, utm, uxm, uxp = u_st

    u_t = (u - utm) / dt
    u_xm = (u - uxm) / dx
    u_xp = (uxp - u) / dx

    uf_st = stencil_var('u', freeze=True)
    uf, _, ufxm, ufxp = uf_st
    ufxmh = (uf + ufxm) * 0.5
    ufxph = (uf + ufxp) * 0.5

    # Conductivity.
    if args.infer_k:
        km = knet_to_k(ctx.neural_net('k', ufxmh)(0))
        kp = knet_to_k(ctx.neural_net('k', ufxph)(0))
    else:
        km = get_ref_k(ufxmh, mod=mod)
        kp = get_ref_k(ufxph, mod=mod)

    # Heat equation.
    qm = u_xm * km
    qp = u_xp * kp
    q_x = (qp - qm) / dx
    fu = u_t - q_x
    res = [fu]

    global mask_imp, u_imp, n_imp
    if n_imp:
        u = u_st[0]
        # Rescale weight to the total number of points.
        coeff = args.kimp * np.prod(domain.shape) / n_imp
        fuimp = mask_imp * (u - u_imp) * coeff
        res += [fuimp]

    # Regularization.
    anneal_factor = get_anneal_factor(ctx.epoch, args.anneal_half)
    if args.kxreg:
        fxreg = u_xm * args.kxreg * anneal_factor
        res += [fxreg]

    if args.ktreg:
        ftreg = u_t * args.ktreg * anneal_factor
        res += [ftreg]

    if args.kw and args.infer_k:
        ww = ctx.state_weights['k']
        ww = mod.concat([mod.reshape(w, [-1]) for w in ww], 0)
        kw = args.kw * anneal_factor
        res += [(tf.stop_gradient(ww) - ww) * kw]
    return res


def state_to_field(key, nnw, state):
    if key in state.weights:
        uw = np.array(state.weights[key][1])
        u = MultigridDecomp.weights_to_cell_field(uw, nnw, tf)
    else:
        u = state.fields[key]
    return np.array(u)


def operator_pinn(mod, ctx):
    global t_in, x_in
    global t_bound, x_bound, u_bound
    global t_init, x_init, u_init
    global t_imp, x_imp, n_imp, u_imp, i_imp
    # Inner points.
    net = ctx.neural_net('u', t_in, x_in)
    u = net(0, 0)
    u_t = net(1, 0)
    u_x = net(0, 1)

    # Conductivity.
    if args.infer_k:
        k = knet_to_k(ctx.neural_net('k', u)(0))
    else:
        k = get_ref_k(u, mod=mod)
    q = k * u_x
    q_x = tf.gradients(q, net.inputs[1])[0]

    # Heat equation.
    fu = u_t - q_x

    # Boundary conditions.
    net = ctx.neural_net('u', t_bound, x_bound)
    fb = net(0, 0) - u_bound

    # Initial conditions.
    net = ctx.neural_net('u', t_init, x_init)
    fi = net(0, 0) - u_init

    res = [fu, fb, fi]

    # Imposed points.
    global mask_imp, u_imp, n_imp, i_imp
    if n_imp:
        net = ctx.neural_net('u', t_imp, x_imp)
        fimp = (net(0, 0) - u_imp.flatten()[i_imp]) * args.kimp
        res += [fimp]

    return res


def get_indices_imposed(i_diag):
    if args.imposed == 'random':
        i_imposed = i_diag.flatten()
        nimp = min(args.nimp, np.prod(i_imposed.size))
        i_imposed = np.random.choice(i_imposed, size=nimp, replace=False)
    elif args.imposed == 'none':
        i_imposed = []
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
    tt = []
    xx = []
    if len(i_imposed):
        res[i_imposed] = 1
        tt = domain.cell_center_by_dim(0).flatten()
        xx = domain.cell_center_by_dim(1).flatten()
        tt = tt[i_imposed]
        xx = xx[i_imposed]
    res = res.reshape(domain.shape)
    return res, tt, xx, i_imposed


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Nt', type=int, default=64, help="Grid size in t")
    parser.add_argument('--Nx', type=int, default=64, help="Grid size in x")
    parser.add_argument('--mg', type=int, default=0, help="Use multigrid")
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
                        choices=('pinn', 'odil'),
                        default='odil',
                        help="Grid size in x")
    parser.add_argument('--infer_k',
                        type=int,
                        default=0,
                        help="Infer conductivity")
    parser.add_argument('--kxreg',
                        type=float,
                        default=0,
                        help="Space regularization weight")
    parser.add_argument('--ktreg',
                        type=float,
                        default=0,
                        help="Time regularization weight")
    parser.add_argument('--kw',
                        type=float,
                        default=0,
                        help="Regularization of neural network weights")
    parser.add_argument('--anneal_half',
                        type=float,
                        default=1000,
                        help="Number of epochs before halving "
                        "of regularization factors")
    parser.add_argument('--kimp',
                        type=float,
                        default=1,
                        help="Weight of imposed points")
    parser.add_argument('--ref_path',
                        type=str,
                        help="Path to reference solution *.pickle")
    parser.add_argument('--imposed',
                        type=str,
                        choices=['random', 'none'],
                        default='none',
                        help="Set of points for imposed solution")
    parser.add_argument('--nimp',
                        type=int,
                        default=500,
                        help="Number of points for imposed=random")
    parser.add_argument('--noise',
                        type=float,
                        default=0,
                        help="Magnitude of perturbation of reference solution")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(nlvl=10)
    parser.set_defaults(outdir='out_heat')
    parser.set_defaults(checkpoint_every=0)
    parser.set_defaults(linsolver='multigrid')
    parser.set_defaults(optimizer='lbfgsb')
    parser.set_defaults(history_full=50)
    return parser.parse_args()


@tf.function()
def u_deriv(weights, tt, xx, nt, nx):
    net_u = odil.util_op.NeuralNet(weights, tt, xx)
    return net_u(nt, nx)


def plot(state, epoch, frame, ref_u):
    global domain
    global t_imp, x_imp

    title0 = "u epoch={:}".format(epoch) if args.plot_title else None
    title1 = "k epoch={:}".format(epoch) if args.plot_title else None
    path0 = "u_{:05d}.{}".format(frame, args.plotext)
    path1 = "k_{:05d}.{}".format(frame, args.plotext)
    printlog(path0, path1)

    if args.solver == 'odil':
        state_u = state_to_field('u', problem.nnw, state)
    elif args.solver == 'pinn':
        tt, xx = domain.cell_center_all()
        eval_u = lambda nt, nx: u_deriv(  #
            state.weights['u'], tt, xx, nt, nx).numpy()
        state_u = eval_u(0, 0)

    odil.util.plot(domain,
                   ref_u,
                   state_u,
                   path=path0,
                   title=title0,
                   cmap='viridis',
                   nslices=5,
                   transpose=True,
                   x_imp=t_imp,
                   y_imp=x_imp,
                   umin=0,
                   umax=1)

    # Plot conductivity.
    fig, ax = plt.subplots(figsize=(1.7, 1.5))
    uk = np.linspace(0, 1, 200)
    k_ref = get_ref_k(uk)
    if args.infer_k:
        k = odil.util_op.eval_neural_net(state.weights['k'], uk).numpy()
        k = knet_to_k(k)
    else:
        k = None
    if k is not None:
        ax.plot(uk, k, zorder=10)
    ax.plot(uk, k_ref, c='C2', lw=1.5, zorder=1)
    ax.set_xlabel('u')
    ax.set_ylabel('k')
    ax.set_ylim(0, 0.03)
    ax.set_title(title1)
    fig.savefig(path1, bbox_inches='tight')
    plt.close(fig)

    if args.dump_data:
        path = "data_{:05d}.pickle".format(frame)
        s = dict()
        s['u'] = state_u
        s['u_ref'] = ref_u
        s['uk'] = uk
        s['k'] = k
        s['k_ref'] = k_ref
        s['t_imp'] = t_imp
        s['x_imp'] = x_imp
        s['i_imp'] = i_imp
        s['domain'] = domain.get_minimal()
        with open(path, 'wb') as f:
            pickle.dump(s, f)


def callback(packed, epoch, dhistory=None, opt=None):
    tstart = time.time()

    def callback_update_time():
        nonlocal tstart
        global g_time_callback
        t = time.time()
        g_time_callback += t - tstart
        tstart = t

    global frame, domain
    global history
    global g_time_callback, g_time_start
    global ref_u

    report = (epoch % args.report_every == 0)
    skip_history = epoch == args.epoch_start and args.epoch_start > 0
    if skip_history:
        printlog("Skipping history after checkpoint")
    calc = (epoch % args.report_every == 0 or epoch % args.history_every == 0
            or epoch < args.history_full
            or (epoch % args.plot_every == 0 and (epoch or args.frames)))
    if calc:
        state = problem.unpack_state(packed)
        memusage = odil.util.get_memory_usage_kb()

    if report:
        printlog("epoch={:05d}".format(epoch))
        if opt.last_residual is not None:
            printlog('residual: ' + ', '.join('{:.5g}'.format(r)
                                              for r in opt.last_residual))
    if args.checkpoint_every and epoch % args.checkpoint_every == 0:
        path = "state_{:06d}.pickle".format(epoch)
        tpath = "state_{:06d}_train.pickle".format(epoch)
        odil.util_op.checkpoint_save(state, path)
        history.save(tpath)
        printlog(path, tpath)
    if epoch % args.plot_every == 0 and (epoch or args.frames):
        plot(state, epoch, frame, ref_u)
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
        if args.solver == 'odil':
            state_u = state_to_field('u', problem.nnw, state)
        elif args.solver == 'pinn':
            tt, xx = domain.cell_center_all()
            state_u = odil.util_op.eval_neural_net(  #
                state.weights['u'], tt, xx).numpy()

        ulin = np.linspace(0, 1)
        k_ref = get_ref_k(ulin)
        if 'k' in state.weights:
            k = odil.util_op.eval_neural_net(state.weights['k'], ulin).numpy()
        else:
            k = k_ref
        k = knet_to_k(k)
        k_err = k - k_ref
        k_err_rel = k_err / max(k_ref)

    if report:
        printlog("u={:.05g}".format(np.max(np.abs(state_u))))
        printlog("memory: {:} MiB".format(memusage // 1024))
        printlog("walltime: {:.3f} s".format(time.time() - g_time_start))

    callback_update_time()

    if report:
        printlog()

    if ((epoch % args.history_every == 0 or epoch < args.history_full)
            and history is not None and not skip_history):
        assert calc
        history.append('epoch', epoch)
        history.append('frame', frame)
        history.append('t_linsolver',
                       problem.timer_last.counters.get('linsolver', 0))
        history.append('t_grad',
                       problem.timer_last.counters.get('eval_grad', 0))
        history.append('t_sparse_fields',
                       problem.timer_last.counters.get('sparse_fields', 0))
        history.append('t_sparse_weights',
                       problem.timer_last.counters.get('sparse_weights', 0))

        history.append('ref_du_linf', np.max(abs(ref_u - state_u)))
        history.append('ref_du_l1', np.mean(abs(ref_u - state_u)))
        history.append('ref_du_l2', np.mean((ref_u - state_u)**2)**0.5)

        history.append('k_err_linf', np.max(abs(k_err_rel)))
        history.append('k_err_l1', np.mean(abs(k_err_rel)))
        history.append('k_err_l2', np.mean((k_err_rel)**2)**0.5)

        if opt.last_loss is not None:
            history.append('loss', opt.last_loss)
        if opt.last_residual is not None:
            for i, r in enumerate(np.array(opt.last_residual)):
                history.append('loss{:}'.format(i), r)

        callback_update_time()
        history.append('tt_linsolver',
                       problem.timer_total.counters.get('linsolver', 0))
        history.append('tt_opt', time.time() - g_time_start - g_time_callback)
        history.append('tt_callback', g_time_callback)
        history.append('walltime', time.time() - g_time_start)
        history.append('memory', memusage / 1024)
        if opt:
            history.append('evals', opt.evals)
        else:
            history.append('evals', 0)

        if dhistory is not None:
            history.append_dict(dhistory)

        history.write()
        callback_update_time()


def load_fields_interp(path, domain):
    '''
    Loads fields from file `path` and interpolates them to shape `domain.shape`.
    '''
    from scipy.interpolate import RectBivariateSpline
    state = odil.util_op.State()
    odil.util_op.checkpoint_load(state,
                                 path,
                                 fields_to_load=['u'],
                                 weights_to_load=[])
    state_u = state.fields['u']

    if state_u.shape != domain.shape:
        x1 = np.linspace(0, 1, state_u.shape[0], endpoint=False)
        y1 = np.linspace(0, 1, state_u.shape[1], endpoint=False)
        x1 += (x1[1] - x1[0]) * 0.5
        y1 += (y1[1] - y1[0]) * 0.5
        fu = RectBivariateSpline(x1, y1, state_u)

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
    else:
        uu = state_u
    return uu


def main():
    global args, problem, domain, frame
    global history
    global ref_u
    global init_u
    global t_in, x_in
    global t_init, x_init, u_init
    global t_bound, x_bound, u_bound
    global t_imp, x_imp, n_imp, u_imp, mask_imp, i_imp

    args = parse_args()

    # Replace with absolute paths before changing current directory.
    if args.checkpoint is not None:
        args.checkpoint = os.path.relpath(args.checkpoint, start=args.outdir)
    if args.checkpoint_train is not None:
        args.checkpoint_train = os.path.relpath(args.checkpoint_train,
                                                start=args.outdir)
    if args.ref_path is not None:
        args.ref_path = os.path.relpath(args.ref_path, start=args.outdir)

    # Change current directory to output directory.
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    odil.util.set_log_file(
        open("train.log", 'w' if args.checkpoint is None else 'a'))

    printlog(' '.join(sys.argv))

    if args.history_every:
        history = odil.History(csvpath='train.csv', warmup=1)
    else:
        history = None

    # Update arguments.
    args.nlvl = min(args.nlvl, int(round(np.log2(args.Nx))))
    args.plot_every *= args.every_factor
    args.history_every *= args.every_factor
    args.report_every *= args.every_factor
    args.checkpoint_every *= args.every_factor
    if args.epochs is None:
        args.epochs = args.frames * args.plot_every

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    fieldnames = []
    neuralnets = dict()
    if args.infer_k:
        neuralnets['k'] = [1] + [5, 5] + [1]  # Conductivity.
    if args.solver == 'odil':
        nw0 = [args.Nt, args.Nx]
        nnw = [[n >> level for n in nw0] for level in range(args.nlvl)]
        printlog('levels', *nnw)
        if args.mg:
            netsize = sum([np.prod(nw) for nw in nnw])
            neuralnets['u'] = [0, netsize]
        else:
            fieldnames.append('u')  # Temperature.
        operator = operator_odil
    elif args.solver == 'pinn':
        neuralnets['u'] = [2] + args.arch + [1]  # Temperature.
        operator = operator_pinn
    else:
        assert False

    domain = odil.util_op.Domain(ndim=2,
                                 shape=(args.Nt, args.Nx),
                                 lower=(0, 0),
                                 upper=(1, 1),
                                 varnames=('t', 'x'),
                                 fieldnames=fieldnames,
                                 neuralnets=neuralnets)

    # Evaluate exact solution, boundary and initial conditions.
    t, x = domain.cell_center_all()
    t1 = domain.cell_center_1d(0)
    x1 = domain.cell_center_1d(1)
    init_u = get_init_u(x1 * 0, x1)

    # Load reference solution.
    if args.ref_path is not None:
        printlog("Loading reference solution from '{}'".format(args.ref_path))
        ref_u = load_fields_interp(args.ref_path, domain)
    else:
        ref_u = get_init_u(t, x)

    mask_imp, t_imp, x_imp, i_imp = get_mask_imposed(domain)
    n_imp = len(x_imp)

    # Add noise after choosing points with imposed values.
    if args.noise:
        ref_u += np.random.uniform(0, args.noise, ref_u.shape)

    u_imp = ref_u

    with open("imposed.csv", 'w') as f:
        f.write('t,x\n')
        for i in range(n_imp):
            f.write('{:},{:}\n'.format(t_imp[i], x_imp[i]))

    if args.solver == 'pinn':
        t_in, x_in = domain.random_inner(args.Nci)
        t_bound, x_bound = domain.random_boundary(1, 0, args.Ncb)
        t_bound2, x_bound2 = domain.random_boundary(1, 1, args.Ncb)
        t_bound = np.hstack((t_bound, t_bound2))
        x_bound = np.hstack((x_bound, x_bound2))
        t_init, x_init = domain.random_boundary(0, 0, args.Ncb)
        u_init = get_init_u(t_init, x_init)
        u_bound = get_init_u(t_bound, x_bound)
        printlog('Number of collocation points:')
        printlog('inside: {:}'.format(len(t_in)))
        printlog('init: {:}'.format(len(t_init)))
        printlog('bound: {:}'.format(len(t_bound)))

    problem = odil.util_op.Problem(operator, domain)

    state = odil.util_op.State()
    frame = 0

    if args.checkpoint is not None:
        printlog("Loading checkpoint '{}'".format(args.checkpoint))
        odil.util_op.checkpoint_load(state,
                                     args.checkpoint,
                                     fields_to_load=domain.fieldnames,
                                     weights_to_load=domain.neuralnets.keys())
        tpath = os.path.splitext(args.checkpoint)[0] + '_train.pickle'
        if args.checkpoint_train is None:
            assert os.path.isfile(tpath), "File not found '{}'".format(tpath)
            args.checkpoint_train = tpath

    if args.checkpoint_train is not None:
        printlog("Loading history from '{}'".format(args.checkpoint_train))
        history.load(args.checkpoint_train)
        args.epoch_start = history.get('epoch', [args.epoch_start])[-1]
        frame = history.get('frame', [args.frame_start])[-1]
        printlog("Starting from epoch={:} frame={:}".format(
            args.epoch_start, args.frame_start))

    problem.init_missing(state)

    if args.solver == 'odil':
        problem.nnw = nnw

    args.ntrainable = len(problem.pack_state(state))
    with open('args.json', 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    odil.util.optimize(args, args.optimizer, problem, state, callback)

    with open('done', 'w') as f:
        pass


if __name__ == "__main__":
    main()
