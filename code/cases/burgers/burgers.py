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
    return (1 - np.cos(x * 6 * np.pi)) * 0.5


def solve_direct(domain, init_u, nsub=1, mod=np):
    """
    Solves with finite differences.
    """
    res_u = mod.zeros(domain.shape)
    ix = domain.cell_index('x')[0]
    dt = domain.step('t')
    dx = domain.step('x')
    nt = domain.size('t')
    nx = domain.size('x')
    zero = 0
    dtsub = dt / nsub

    u = init_u
    res_u[0] = u

    extrap = odil.util_op.extrap_linear
    for it in range(1, nt):
        for sub in range(nsub):
            um = mod.roll(u, 1)
            up = mod.roll(u, -1)
            # Zero Dirichlet conditions.
            um = mod.where(ix == 0, extrap(up, u), um)
            up = mod.where(ix == nx - 1, extrap(um, u), up)
            umh = (u + um) * 0.5
            uph = (u + up) * 0.5
            # Fluxes.
            qm = 0.5 * mod.where(umh > 0, um, mod.where(umh < 0, u, umh))**2
            qp = 0.5 * mod.where(uph > 0, u, mod.where(uph < 0, up, uph))**2
            qm = mod.where((ix == 0) & (umh > 0), zero, qm)
            qp = mod.where((ix == nx - 1) & (uph < 0), zero, qp)
            q_x = (qp - qm) / dx
            u_t = -q_x
            u += u_t * dtsub
        res_u[it] = u
    return res_u


def get_anneal_factor(epoch, anneal_half):
    return 0.5**(epoch / anneal_half) if anneal_half else 1


def operator_odil(mod, ctx):
    global args, domain
    global init_u
    dt = ctx.step('t')
    dx = ctx.step('x')
    x = ctx.cell_center('x')
    zero = mod.cast(0, domain.dtype)
    it = ctx.cell_index('t')
    ix = ctx.cell_index('x')
    nt = ctx.size('t')
    nx = ctx.size('x')

    def stencil_var(key, freeze=False, shift_t=0):
        if args.mg:
            qw = ctx.neural_net(key)()
            q = MultigridDecomp.weights_to_field(qw,
                                                 problem.nnw,
                                                 mod,
                                                 cell=True)
            q = mod.roll(q, -shift_t, 0)
            st = [
                q,
                mod.roll(q, 1, 1),
                mod.roll(q, -1, 1),
            ]
        else:
            st = [
                ctx.field(key, shift_t, 0, freeze=freeze),
                ctx.field(key, shift_t, -1, freeze=freeze),
                ctx.field(key, shift_t, 1, freeze=freeze),
            ]
        return st

    def apply_bc_u(st):
        extrap = odil.util_op.extrap_linear
        # Zero Dirichlet conditions.
        st[1] = mod.where(ix == 0, extrap(st[2], st[0]), st[1])
        st[2] = mod.where(ix == nx - 1, extrap(st[1], st[0]), st[2])
        return st

    u_st = stencil_var('u')
    utp = stencil_var('u', shift_t=1)[0]
    apply_bc_u(u_st)

    u, um, up = u_st

    umh = (u + um) * 0.5
    uph = (u + up) * 0.5
    # Fluxes.
    qm = 0.5 * mod.where(umh > 0, um, mod.where(umh < 0, u, umh))**2
    qp = 0.5 * mod.where(uph > 0, u, mod.where(uph < 0, up, uph))**2
    qm = mod.where((ix == 0) & (umh > 0), zero, qm)
    qp = mod.where((ix == nx - 1) & (uph < 0), zero, qp)
    q_x = (qp - qm) / dx
    u_t = (utp - u) / dt

    # Burgers equation.
    fu = u_t + q_x
    fu = mod.where(it == nt - 1, zero, fu)
    if not args.bc:
        # Do not impose the equation in boundary cells.
        fu = mod.where((ix == 0) | (ix == nx - 1), zero, fu)
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
        u_x = (u - um) / dx
        u_x = mod.where(ix == 0, zero, u_x)
        fxreg = u_x * args.kxreg * anneal_factor
        res += [fxreg]

    if args.ktreg:
        ftreg = u_t * args.ktreg * anneal_factor
        ftreg = mod.where(it == nt - 1, zero, ftreg)
        res += [ftreg]

    # Damping.
    if not args.mg and args.urelax:
        # Add pressure relaxation to define pressure inside the body,
        # needed for Newton's method to have non-zero diagonal for pressure.
        res += [(ctx.field('u') - ctx.field('u', freeze=True)) * args.urelax]
    return res


def state_to_field(key, nnw, state):
    if key in state.weights:
        uw = np.array(state.weights[key][1])
        u = MultigridDecomp.weights_to_field(uw, nnw, tf, cell=True)
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


def get_indices_imposed(domain, i_diag):
    nt, nx = domain.shape
    if args.imposed == 'random':
        i_imposed = i_diag.flatten()
        nimp = min(args.nimp, np.prod(i_imposed.size))
        i_imposed = np.random.choice(i_imposed, size=nimp, replace=False)
    elif args.imposed == 'init':
        i_imposed = i_diag[0]
    elif args.imposed in ['rect', 'recth']:
        it0 = nt // 3
        it1 = nt * 2 // 3 + 1
        ix0 = nx // 3
        ix1 = nx * 2 // 3 + 1
        if args.imposed == 'recth':
            i_imposed = np.hstack([
                i_diag[it0, slice(ix0, ix1)],
                i_diag[it1 - 1, slice(ix0, ix1)],
                i_diag[slice(it0, it1), ix0],
                i_diag[slice(it0, it1), ix1 - 1],
            ])
        else:
            i_imposed = i_diag[slice(it0, it1), slice(ix0, ix1)]

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
    i_imposed = get_indices_imposed(domain, i_diag)
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
    parser.add_argument('--Nx', type=int, default=64, help="Grid size in x")
    parser.add_argument('--Nt', type=int, default=None, help="Grid size in t")
    parser.add_argument('--mg', type=int, default=1, help="Use multigrid")
    parser.add_argument('--urelax',
                        type=float,
                        default=0,
                        help="Relaxation coefficient")
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
                        choices=('pinn', 'odil', 'direct'),
                        default='odil',
                        help="Grid size in x")
    parser.add_argument('--tmax', type=float, default=1, help="Maximum time")
    parser.add_argument('--kxreg',
                        type=float,
                        default=0,
                        help="Space regularization weight")
    parser.add_argument('--ktreg',
                        type=float,
                        default=0,
                        help="Time regularization weight")
    parser.add_argument('--anneal_half',
                        type=float,
                        default=3,
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
                        #choices=['random', 'init', 'none'],
                        default='none',
                        help="Set of points for imposed solution")
    parser.add_argument('--nimp',
                        type=int,
                        default=500,
                        help="Number of points for imposed=random")
    parser.add_argument('--bc',
                        type=int,
                        default=0,
                        help="Impose boundary conditions")
    parser.add_argument('--noise',
                        type=float,
                        default=0,
                        help="Magnitude of perturbation of reference solution")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(nlvl=10)
    parser.set_defaults(outdir='out_burgers')
    parser.set_defaults(checkpoint_every=0)
    parser.set_defaults(linsolver='multigrid')
    parser.set_defaults(optimizer='lbfgsb')
    parser.set_defaults(history_full=50)
    return parser.parse_args()


@tf.function()
def u_deriv(weights, tt, xx, nt, nx):
    net_u = odil.util_op.NeuralNet(weights, tt, xx)
    return net_u(nt, nx)


def plot(state, epoch, frame, ref_u, reforig_u=None):
    global domain
    global t_imp, x_imp

    title0 = "u epoch={:}".format(epoch) if args.plot_title else None
    path0 = "u_{:05d}.{}".format(frame, args.plotext)
    printlog(path0)

    if args.solver in ['odil', 'direct']:
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
                   x_imp=x_imp,
                   y_imp=t_imp,
                   umin=0,
                   umax=1)

    if args.dump_data:
        path = "data_{:05d}.pickle".format(frame)
        s = dict()
        s['u'] = state_u
        s['u_ref'] = ref_u
        s['u_reforig'] = reforig_u
        s['t_imp'] = t_imp
        s['x_imp'] = x_imp
        s['i_imp'] = i_imp
        s['domain'] = domain.get_minimal()
        with open(path, 'wb') as f:
            pickle.dump(s, f)


def callback(packed, epoch, dhistory=None, opt=None):
    global frame, domain
    global history
    global g_time_callback, g_time_start
    global ref_u, reforig_u

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
        plot(state, epoch, frame, ref_u, reforig_u)
        frame += 1

    if calc:
        if args.solver in ['odil', 'direct']:
            state_u = state_to_field('u', problem.nnw, state)
        elif args.solver == 'pinn':
            tt, xx = domain.cell_center_all()
            state_u = odil.util_op.eval_neural_net(  #
                state.weights['u'], tt, xx).numpy()

    if report:
        printlog("u={:.05g}".format(np.max(np.abs(state_u))))
        printlog("memory: {:} MiB".format(memusage // 1024))
        printlog("walltime: {:.3f} s".format(time.time() - g_time_start))

    if report:
        printlog()

    if ((epoch % args.history_every == 0 or epoch < args.history_full)
            and history is not None and not skip_history):
        assert calc
        history.append('epoch', epoch)
        history.append('frame', frame)

        history.append('u_err_inf', np.max(abs(ref_u - state_u)))
        history.append('u_err_l1', np.mean(abs(ref_u - state_u)))
        history.append('u_err_l2', np.mean((ref_u - state_u)**2)**0.5)

        if opt.last_loss is not None:
            history.append('loss', opt.last_loss)
        if opt.last_residual is not None:
            for i, r in enumerate(np.array(opt.last_residual)):
                history.append('loss{:}'.format(i), r)

        history.append('walltime', time.time() - g_time_start)
        history.append('memory', memusage / 1024)
        if opt:
            history.append('evals', opt.evals)
        else:
            history.append('evals', 0)

        history.write()


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
    global ref_u, reforig_u
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
    if args.Nt is None:
        args.Nt = args.Nx
    if args.solver == 'direct':
        args.mg = 0
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
    if args.solver in ['odil', 'direct']:
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
                                 upper=(args.tmax, 1),
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
    reforig_u = ref_u
    if args.noise:
        ref_u = ref_u + np.random.uniform(0, args.noise, ref_u.shape)

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

    if args.solver in ['odil', 'direct']:
        problem.nnw = nnw

    args.ntrainable = len(problem.pack_state(state))
    with open('args.json', 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    if args.solver == 'direct':
        opt = odil.optimizer.Optimizer()
        frame = 0
        packed = problem.pack_state(state)
        callback(packed, 0, opt=opt)

        u = solve_direct(domain, init_u)
        state.fields['u'] = u

        frame = 1
        packed = problem.pack_state(state)
        callback(packed, 1, opt=opt)
    else:
        odil.util.optimize(args, args.optimizer, problem, state, callback)

    with open('done', 'w') as f:
        pass


if __name__ == "__main__":
    main()
