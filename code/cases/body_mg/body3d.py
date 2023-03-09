#!/usr/bin/env python3

import argparse
import json
import numpy as np
import os
import sys
import time
from functools import partial
import odil
from odil import tf
from odil import MultigridDecomp
import matplotlib.pyplot as plt

g_time_start = time.time()
printlog = odil.util.printlog


def get_body_chi(body, domain, bx, by, bz, br):
    h = domain.step_by_dim(0)
    if body == 'circle':
        if domain.ndim == 3:
            x, y, z = domain.cell_center_all()
        elif domain.ndim == 2:
            x, y = domain.cell_center_all()
            z = bz
        else:
            assert False
        r = np.sqrt((x - bx)**2 + (y - by)**2 + (z - bz)**2)
        dr = br - r
        res = np.clip(0.5 + dr / h, 0, 1)
    elif body == 'flat':
        phi = np.pi / 4
        brx = br * 1.25
        bry = brx / 3
        qxx = np.cos(phi)
        qxy = -np.sin(phi)
        qyy = np.cos(phi)
        qyx = np.sin(phi)

        if domain.ndim == 3:
            x, y, z = domain.cell_center_all()
        elif domain.ndim == 2:
            x, y = domain.cell_center_all()
            z = bz
        else:
            assert False
        qx = x - bx
        qy = y - by
        qz = z - bz
        qx, qy = qxx * qx + qxy * qy, qyx * qx + qyy * qy
        r = np.sqrt((qx / brx)**2 + (qy / bry)**2 + (qz / br)**2) * br
        dr = br - r
        res = np.clip(0.5 + dr / h, 0, 1)
    elif body == 'half':
        phi = 0
        brx = br
        bry = br
        qxx = np.cos(phi)
        qxy = -np.sin(phi)
        qyy = np.cos(phi)
        qyx = np.sin(phi)

        if domain.ndim == 3:
            x, y, z = domain.cell_center_all()
        elif domain.ndim == 2:
            x, y = domain.cell_center_all()
            z = bz
        else:
            assert False
        qx = x - bx
        qy = y - by
        qz = z - bz
        qx, qy = qxx * qx + qxy * qy, qyx * qx + qyy * qy
        r = np.sqrt((qx / brx)**2 + (qy / bry)**2 + (qz / br)**2) * br
        dr = br - r
        dr = np.minimum(dr, -qy)
        res = np.clip(0.5 + dr / h, 0, 1)
    else:
        assert False
    res = res.astype(domain.dtype)
    return res


def get_anneal_factor(epoch, anneal_half):
    return 0.5**(epoch / anneal_half) if anneal_half else 1

def transform_chi(chi, mod):
    if isinstance(chi, list):
        return [transform_chi(c, mod) for c in chi]
    return 1 / (1 + mod.exp(-chi + 5))


def operator_ns(mod, ctx, args=None):
    global problem
    dim = problem.domain.ndim
    dirs = range(dim)
    varnames = problem.domain.varnames
    ones = ctx.field('ones')
    zeros = ctx.field('zeros')
    dw = [ctx.step(n) for n in varnames]
    iw = [ctx.cell_index(n) for n in varnames]
    nw = [ctx.size(n) for n in varnames]
    epoch = ctx.epoch

    def stencil_var(key, freeze=False):
        if args.mg:
            qw = ctx.neural_net(key)()
            if freeze and args.wfreeze:
                qw = tf.stop_gradient(qw)
            q = MultigridDecomp.weights_to_node_field(qw,
                                                      problem.nnw,
                                                      mod,
                                                      method=args.mg_interp)
            st = [None] * (2 * dim + 1)
            st[0] = q
            for i in dirs:
                st[2 * i + 1] = mod.roll(q, 1, i)
                st[2 * i + 2] = mod.roll(q, -1, i)
        else:
            st = [ctx.field(key, freeze=freeze)]
            for i in dirs:
                w = [-1 if j == i else 0 for j in dirs]
                st.append(ctx.field(key, *w, freeze=freeze))
                w = [1 if j == i else 0 for j in dirs]
                st.append(ctx.field(key, *w, freeze=freeze))
        return st

    def split_wm_wp(st):
        q = st[0]
        qwm = [st[2 * i + 1] for i in dirs]
        qwp = [st[2 * i + 2] for i in dirs]
        return q, qwm, qwp

    def split_wmm_wpp(st5):
        qwmm = [st5[2 * i] for i in dirs]
        qwpp = [st5[2 * i + 1] for i in dirs]
        return qwmm, qwpp

    def central(st):
        _, qwm, qwp = split_wm_wp(st)
        return [(qwp[i] - qwm[i]) / (2 * dw[i]) for i in dirs]

    def grad_upwind_second(st, st5, vw):
        'Gradient using second-order upwind scheme.'
        q, qwm, qwp = split_wm_wp(st)
        qwmm, qwpp = split_wmm_wpp(st5)
        q_wm = [(qwmm[i] - 4 * qwm[i] + 3 * q) / (2 * dw[i]) for i in dirs]
        q_wp = [(-3 * q + 4 * qwp[i] - qwpp[i]) / (2 * dw[i]) for i in dirs]
        q_w = [mod.where(vw[i] > 0, q_wm[i], q_wp[i]) for i in dirs]
        return q_w

    def grad_upwind_high(st, st5, vw):
        'Gradient using a high-order upwind scheme.'
        return grad_upwind_second(st, st5, vw)

    def grad_upwind_low(st, vw):
        'Gradient using a low-order upwind scheme.'
        q, qwm, qwp = split_wm_wp(st)
        q_wm = [(q - qwm[i]) / dw[i] for i in dirs]
        q_wp = [(qwp[i] - q) / dw[i] for i in dirs]
        q_w = [mod.where(vw[i] > 0, q_wm[i], q_wp[i]) for i in dirs]
        return q_w

    def laplace(st):
        q, qwm, qwp = split_wm_wp(st)
        q_ww = [(qwp[i] - 2 * q + qwm[i]) / dw[i]**2 for i in dirs]
        q_lap = sum(q_ww)
        return q_lap

    def apply_bc(st, bcvalues):
        'Quadratic extrapolation with boundary conditions.'
        q, qwm, qwp = split_wm_wp(st)

        def extrap(u0, u1, u15):
            if u15 == 'extrap':
                return odil.util_op.extrap_linear(u0, u1)
            return odil.util_op.extrap_quadh(u0, u1, u15)

        for i in dirs:
            qm = mod.where(
                iw[i] == 0,  #
                extrap(qwp[i], q, bcvalues[2 * i]),
                qwm[i])
            qp = mod.where(
                iw[i] == nw[i] - 1,  #
                extrap(qwm[i], q, bcvalues[2 * i + 1]),
                qwp[i])
            qwm[i], qwp[i] = qm, qp
        for i in dirs:
            st[2 * i + 1] = qwm[i]
            st[2 * i + 2] = qwp[i]

    def apply_bc_extrap(st):
        'Linear extrapolation from inner cells to halo cells.'
        q, qwm, qwp = split_wm_wp(st)
        extrap = odil.util_op.extrap_linear
        for i in dirs:
            qm = mod.where(iw[i] == 0, extrap(qwp[i], q), qwm[i])
            qp = mod.where(iw[i] == nw[i] - 1, extrap(qwm[i], q), qwp[i])
            qwm[i], qwp[i] = qm, qp
        for i in dirs:
            st[2 * i + 1] = qwm[i]
            st[2 * i + 2] = qwp[i]

    def stencil(q):
        qq = [None] * (2 * dim)
        for i in dirs:
            qq[2 * i] = mod.roll(q, shift=1, axis=i)
            qq[2 * i + 1] = mod.roll(q, shift=-1, axis=i)
        return [q] + qq

    def stencil5(st):
        q, *qq = st
        q5 = [None] * (2 * dim)
        for i in dirs:
            q5[2 * i] = mod.roll(qq[2 * i], shift=1, axis=i)
            q5[2 * i + 1] = mod.roll(qq[2 * i + 1], shift=-1, axis=i)
        # Second-order extrapolation near boundaries.
        for i in dirs:
            ext = odil.util_op.extrap_quad
            q5[2 * i] = mod.where(
                iw[i] == 0,
                ext(qq[2 * i + 1], q, qq[2 * i]),
                q5[2 * i],
            )
            q5[2 * i + 1] = mod.where(
                iw[i] == nw[i] - 1,
                ext(qq[2 * i], q, qq[2 * i + 1]),
                q5[2 * i + 1],
            )
        return q5

    def upwind_mix(vhf, vl, vlf):
        return tuple(hf + l - lf for hf, l, lf in zip(vhf, vl, vlf))

    vw_st = [stencil_var('v' + n) for n in varnames]
    vwf_st = [stencil_var('v' + n, freeze=True) for n in varnames]
    for i in dirs:  # Loop over velocity components.
        bcvalues = ['extrap'] * (2 * dim)
        if i == 0:
            bcvalues[0] = args.inletvx
        elif i == 1:
            bcvalues[0] = 0
            bcvalues[2] = 0
            bcvalues[3] = 0
        elif i == 2:
            bcvalues[0] = 0
            bcvalues[4] = 0
            bcvalues[5] = 0
        apply_bc(vw_st[i], bcvalues)
        apply_bc(vwf_st[i], bcvalues)
    vwf_st5 = [stencil5(vwf_st[i]) for i in dirs]
    vw = [vw_st[i][0] for i in dirs]
    vwf = [vwf_st[i][0] for i in dirs]

    p_st = stencil_var('p')
    apply_bc_extrap(p_st)

    # vw_w[i][j] is a derivative of v_i wrt x_j.
    vw_w = [
        upwind_mix(grad_upwind_high(vwf_st[i], vwf_st5[i], vwf),
                   grad_upwind_low(vw_st[i], vwf),
                   grad_upwind_low(vwf_st[i], vwf)) for i in dirs
    ]

    p_wc = central(p_st)
    vw_lap = [laplace(vw_st[i]) for i in dirs]

    mu = args.mu

    if args.infer_chi:
        chi_st = stencil_var('chi')
        chif_st = stencil_var('chi', freeze=True)
        chi_st = transform_chi(chi_st, mod=mod)
        chif_st = transform_chi(chif_st, mod=mod)
        # Zero Dirichlet boundary conditions for the body fraction.
        apply_bc(chi_st, [0] * (2 * dim))
        chi = chi_st[0]
        chif = chif_st[0]
    else:
        chi = problem.body_chi
        chif = problem.body_chi

    # Momentum equations.
    fvw = [
        sum(vwf[j] * vw_w[i][j] for j in dirs) - mu * vw_lap[i] + p_wc[i]
        for i in dirs
    ]
    # Add penalization term.
    lamb = args.lamb
    if args.blend:
        for i in dirs:
            fvw[i] = fvw[i] * (1 - chi) + vw[i] * chi * lamb
    else:
        for i in dirs:
            fvw[i] += vw[i] * chi * lamb / dw[0]
    res = fvw

    if args.rhiechow:
        # Rhie-Chow correction to remove pressure oscillation.
        '''
        Diagonal part of momentum equations:
            vw = - b p_w
            g * vw + p_w = 0
        '''
        g = sum(mod.abs(vwf[i]) / dw[i] + mu * 2 / dw[i]**2 for i in dirs)
        if args.blend:
            b = 1 / g
            b *= 1 - chif
        else:
            g += lamb * chif / dw[0]
            b = 1 / g
        pf_st = stencil_var('p', freeze=True)
        apply_bc_extrap(pf_st)
        _, bwm, bwp = split_wm_wp(stencil(b))
        p, pwm, pwp = split_wm_wp(p_st)
        pf, pfwm, pfwp = split_wm_wp(pf_st)
        pfwmm, pfwpp = split_wmm_wpp(stencil5(pf_st))

        vwm = [vw_st[i][2 * i + 1] for i in dirs]
        vwp = [vw_st[i][2 * i + 2] for i in dirs]

        rhiechow = args.rhiechow
        qwm = [
            (vw[i] + vwm[i]) * 0.5 + (
                bwm[i] * (pf - pfwmm[i]) / (2 * dw[i]) +  #
                b * (pfwp[i] - pfwm[i]) / (2 * dw[i]) -  #
                (bwm[i] + b) * (p - pwm[i]) / dw[i]) * 0.5 * rhiechow
            for i in dirs
        ]
        qwp = [
            (vw[i] + vwp[i]) * 0.5 + (
                bwp[i] * (pfwpp[i] - pf) / (2 * dw[i]) +  #
                b * (pfwp[i] - pfwm[i]) / (2 * dw[i]) -  #
                (bwp[i] + b) * (pwp[i] - p) / dw[i]) * rhiechow for i in dirs
        ]

        # Extrapolate flux through boundaries.
        for i in dirs:
            qwm[i] = mod.where(  #
                iw[i] == 0, (vw[i] + vwm[i]) * 0.5, qwm[i])
            qwp[i] = mod.where(  #
                iw[i] == nw[i] - 1, (vw[i] + vwp[i]) * 0.5, qwp[i])

        fp = sum((qwp[i] - qwm[i]) / dw[i] for i in dirs)
    else:
        # Continuity equation without correction,
        # will produce oscillatory pressure.
        # `vw_wc[i]` is a derivative of v_i wrt x_i.
        vw_wc = [central(vw_st[i])[i] for i in dirs]
        fp = sum(vw_wc)

    # Zero outlet pressure.
    p = p_st[0]
    fp = mod.where(iw[0] == nw[0] - 1, p, fp)
    res += [fp]
    if not args.mg and args.prelax:
        # Add pressure relaxation to define pressure inside the body,
        # needed for Newton's method to have non-zero diagonal for pressure.
        res += [(ctx.field('p') - ctx.field('p', freeze=True)) * args.prelax]

    # Impose velocity in measurements points.
    if problem.imp_size:
        # Rescale weight to the total number of points.
        coeff = args.kimp * np.prod(problem.domain.shape) / problem.imp_size
        for i in dirs:
            res += [(vw[i] - problem.ref_vw[i]) * problem.imp_mask * coeff]

    # Velocity regularization.
    if args.kreg:
        k = args.kreg * get_anneal_factor(epoch, args.kregdecay)
        for i in dirs:
            res += [laplace(vw_st[i]) * k]

    # Body fraction regularization.
    if args.infer_chi:
        if args.kreg2:
            k2 = args.kreg2 * get_anneal_factor(epoch, args.kreg2decay)
            chi_lapl = laplace(chi_st)
            res += [chi_lapl * k2]

        if args.kreg3:
            dchi = chi * (1 - chi)
            res += [dchi * args.kreg3]

    return res


def get_indices_imposed(domain, args, i_diag):
    if args.imposed == 'random':
        imp_i = i_diag.flatten()
        nimp = min(args.nimp, np.prod(imp_i.size))
        imp_i = np.random.choice(imp_i, size=nimp, replace=False)
    elif args.imposed == 'all':
        imp_i = i_diag.flatten()
    elif args.imposed == 'ring':
        imp_i = i_diag.flatten()
        nimp = min(args.nimp, np.prod(imp_i.size))
        imp_i = np.random.choice(imp_i, size=nimp, replace=False)
        bx, by, bz = args.body_cx, args.body_cy, args.body_cz
        if domain.ndim == 3:
            x = domain.cell_center_by_dim(0).flatten()
            y = domain.cell_center_by_dim(1).flatten()
            z = domain.cell_center_by_dim(2).flatten()
            r = np.sqrt((x - bx)**2 + (y - by)**2 + (z - bz)**2)
        elif domain.ndim == 2:
            x = domain.cell_center_by_dim(0).flatten()
            y = domain.cell_center_by_dim(1).flatten()
            r = np.sqrt((x - bx)**2 + (y - by)**2)
        else:
            assert False
        br = args.body_r * 1.5
        br2 = args.body_r * 3
        imp_i = imp_i[np.where((r[imp_i] > br) & (r[imp_i] < br2))]
    elif args.imposed == 'none':
        imp_i = []
    else:
        raise ValueError("Unknown imposed=" + args.imposed)
    return imp_i


def get_mask_imposed(args, domain):
    size = np.prod(domain.shape)
    Nf = len(domain.fieldnames)
    row = range(size)
    i_diag = np.reshape(row, domain.shape)
    imp_i = get_indices_imposed(domain, args, i_diag)
    imp_i = np.unique(imp_i)
    res = np.zeros(size)
    if len(imp_i):
        res[imp_i] = 1
        points = [
            domain.cell_center_by_dim(i).flatten()[imp_i]
            for i in range(domain.ndim)
        ]
        points = np.array(points).T
    else:
        points = []
    res = res.reshape(domain.shape)
    return res, points, imp_i


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Re', type=float, default=60, help="Reynolds number")
    parser.add_argument('--dim',
                        type=int,
                        default=3,
                        choices=(2, 3),
                        help="Space dimensionality")
    parser.add_argument('--N', type=int, default=65, help="Grid size")
    parser.add_argument('--mg', type=int, default=1, help="Use multigrid")
    parser.add_argument('--rhiechow',
                        type=float,
                        default=0.1,
                        help="Use Rhie-Chow correction to remove "
                        "pressure oscillation")
    parser.add_argument('--prelax',
                        type=float,
                        default=0.01,
                        help="Pressure relaxation coefficient")
    parser.add_argument('--infer_chi',
                        type=int,
                        default=0,
                        help="Infer body shape")
    parser.add_argument('--xlen',
                        type=int,
                        default=2,
                        help="Domain length in x")
    parser.add_argument('--inletvx',
                        type=float,
                        default=1,
                        help="Inlet velocity")
    parser.add_argument('--lamb',
                        type=float,
                        default=1,
                        help="Penalization factor")
    parser.add_argument('--body',
                        type=str,
                        choices=('circle', 'flat', 'half'),
                        default='circle',
                        help="Reference body shape")
    parser.add_argument('--body_r',
                        type=float,
                        default=0.2,
                        help="Body radius")
    parser.add_argument('--body_cx',
                        type=float,
                        default=0.5,
                        help="Body center x")
    parser.add_argument('--body_cy',
                        type=float,
                        default=0.5,
                        help="Body center y")
    parser.add_argument('--body_cz',
                        type=float,
                        default=0.5,
                        help="Body center z")
    parser.add_argument('--blend',
                        type=int,
                        default=1,
                        help="Use blended penalization")
    parser.add_argument('--ref_path',
                        type=str,
                        help="Path to reference solution *.pickle")
    parser.add_argument('--imposed',
                        type=str,
                        choices=['random', 'ring', 'none'],
                        default='none',
                        help="Set of measurement points")
    parser.add_argument('--nimp',
                        type=int,
                        default=500,
                        help="Number of points for imposed=random")
    parser.add_argument('--kimp',
                        type=float,
                        default=1,
                        help="Weight of imposed points")
    parser.add_argument('--kreg',
                        type=float,
                        default=0,
                        help="Laplacian regularization factor for velocity")
    parser.add_argument('--kregdecay',
                        type=float,
                        default=0,
                        help="Laplacian regularization decay with epoch")
    parser.add_argument('--kreg2',
                        type=float,
                        default=0,
                        help="Laplacian factor for chi")
    parser.add_argument('--kreg2decay',
                        type=float,
                        default=0,
                        help="Laplacian factor decay with epoch for chi")
    parser.add_argument('--kreg3',
                        type=float,
                        default=0,
                        help="Factor to enforce chi=0 or chi=1")
    parser.add_argument('--wfreeze',
                        type=int,
                        default=0,
                        help="Respect freeze for weights or if mg=1")

    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(nlvl=10)
    parser.set_defaults(double=0)
    parser.set_defaults(plotext='png')
    parser.set_defaults(frames=10)
    parser.set_defaults(checkpoint_every=0)
    parser.set_defaults(plot_every=2500, report_every=50, history_every=50)
    parser.set_defaults(outdir='out_body3d')
    parser.set_defaults(history_full=50)

    parser.set_defaults(optimizer='lbfgs')

    parser.set_defaults(every_factor=1)
    parser.set_defaults(lr=0.001)
    parser.set_defaults(linsolver='multigrid')
    parser.set_defaults(linsolver_maxiter=10)
    return parser.parse_args()


def state_to_field(key, nnw, state):
    if key in state.weights:
        uw = np.array(state.weights[key][1])
        u = MultigridDecomp.weights_to_node_field(uw,
                                                  nnw,
                                                  tf,
                                                  method=args.mg_interp)
    else:
        u = state.fields[key]
    return np.array(u)


def write_field(u, name, path, domain):
    dim = domain.ndim
    dw = [domain.step_by_dim(i) for i in range(dim)]
    axes = tuple(reversed(range(dim)))
    u = np.transpose(u, axes)
    odil.write_raw_with_xmf(u, path, spacing=dw, name=name)


def plot(problem, state, epoch, frame):
    domain = problem.domain
    dim = domain.ndim
    paths = []
    keys = ['vx', 'vy', 'vz'][:dim] + ['p']
    if 'chi' in domain.neuralnets or 'chi' in domain.fieldnames:
        keys.append('chi')
    for key in keys:
        u = state_to_field(key, problem.nnw, state)
        if key == 'chi':
            u = transform_chi(u, mod=np)
        path = key + '_{:05d}.xmf'.format(frame)
        paths.append(path)
        write_field(u, key, path, domain)
    printlog(' '.join(paths))


def callback(packed, epoch, dhistory=None, opt=None):
    global g_time_start
    global frame, prevreport, history, problem, args
    varnames = problem.domain.varnames

    report = (epoch % args.report_every == 0)
    calc = (epoch % args.report_every == 0 or epoch % args.history_every == 0
            or epoch < args.history_full
            or (epoch % args.plot_every == 0 and (epoch or args.frames)))

    if calc:
        state = problem.unpack_state(packed)
        state_vw = [
            state_to_field('v' + n, problem.nnw, state) for n in varnames
        ]
        state_p = state_to_field('p', problem.nnw, state)
        if args.infer_chi:
            state_chi = state_to_field('chi', problem.nnw, state)
            state_chi = transform_chi(state_chi, np)

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
        ref_vw = problem.ref_vw
        ref_p = problem.ref_p
        ref_chi = problem.ref_chi
        history.append('epoch', epoch)
        history.append('frame', frame)
        if opt.last_loss is not None:
            history.append('loss', opt.last_loss)
        if opt.last_residual is not None:
            for i, r in enumerate(opt.last_residual):
                history.append('loss{:}'.format(i), r)
        history.append('walltime', time.time() - g_time_start)
        history.append('memory', memusage / 1024)
        domain = problem.domain
        dirs = range(domain.ndim)
        fields = [('v' + domain.varnames[i], abs(ref_vw[i] - state_vw[i]))
                  for i in dirs]
        fields.append(('p', abs(ref_p - state_p)))
        if args.infer_chi:
            fields.append(('chi', abs(ref_chi - state_chi)))
        else:
            fields.append(('chi', np.array([0.])))
        for name, dv in fields:
            history.append(name + '_err_linf', np.max(dv))
            history.append(name + '_err_l1', np.mean(dv))
            history.append(name + '_err_l2', np.mean(dv**2)**0.5)
        if dhistory is not None:
            history.append_dict(dhistory)
        history.write()


def load_reference(path, nnw, domain):
    state = odil.util_op.State()
    dim = domain.ndim
    keys = ['vx', 'vy', 'vz'][:dim] + ['p']
    odil.util_op.checkpoint_load(state, args.ref_path)
    if 'vx' in state.weights:
        # Detect if reference had nlvl=1.
        if len(state.weights['vx'][1]) == np.prod(domain.shape):
            nnw = [domain.shape]
    res = dict()
    for key in keys:
        res[key] = state_to_field(key, nnw, state)
    return res


def make_problem(args):
    dim = args.dim
    N = args.N
    nw0 = [(N - 1) * args.xlen + 1, N, N][:dim]
    nnw = [[((n - 1) >> level) + 1 for n in nw0] for level in range(args.nlvl)]
    printlog('levels', *nnw)
    varnames = ['x', 'y', 'z'][:dim]

    if args.mg:
        netsize = sum([np.prod(nw) for nw in nnw])
        neuralnets = {n: [0, netsize] for n in ['vx', 'vy', 'vz'][:dim]}
        neuralnets['p'] = [0, netsize]
        if args.infer_chi:
            neuralnets['chi'] = [0, netsize]
        fieldnames = []
    else:
        fieldnames = ['vx', 'vy', 'vz'][:dim] + ['p']
        if args.infer_chi:
            fieldnames.append('chi')
        neuralnets = {}

    dtype = np.float64 if args.double else np.float32
    domain = odil.util_op.Domain(ndim=dim,
                                 shape=nw0,
                                 dtype=dtype,
                                 varnames=varnames,
                                 upper=[args.xlen, 1, 1][:dim],
                                 fieldnames=fieldnames,
                                 neuralnets=neuralnets)

    # Load reference solution.
    if args.ref_path is not None:
        printlog("Loading reference solution from '{}'".format(args.ref_path))
        ref_fields = load_reference(args.ref_path, nnw, domain)
        ref_vw = [ref_fields[key] for key in ['vx', 'vy', 'vz'][:dim]]
        ref_p = ref_fields['p']
    else:
        ref_vw = [np.zeros(domain.shape) for i in range(dim)]
        ref_p = np.zeros(domain.shape)

    # Generate measurement points to impose reference solution.
    imp_mask, imp_points, imp_indices = get_mask_imposed(args, domain)
    imp_size = len(imp_points)
    with open("imposed.csv", 'w') as f:
        f.write(','.join(domain.varnames) + '\n')
        for p in imp_points:
            f.write(','.join(['{:}'] * dim).format(*p) + '\n')

    op = partial(operator_ns, args=args)
    problem = odil.util_op.Problem(op, domain)
    state = odil.util_op.State()
    problem.init_missing(state)
    # Set initial velocity
    if args.mg:
        ones = MultigridDecomp.get_weights_coarse_ones(nnw, domain.dtype, mod=tf)
        state.weights['vx'][1].assign_add(ones * args.inletvx)
    else:
        ones = tf.ones(domain.shape, dtype=domain.dtype)
        state.fields['vx'].assign_add(ones * args.inletvx)
    problem.ref_vw = ref_vw
    problem.ref_p = ref_p
    problem.nnw = nnw
    problem.body_chi = get_body_chi(args.body,
                                    domain,
                                    bx=args.body_cx,
                                    by=args.body_cy,
                                    bz=args.body_cz,
                                    br=args.body_r)
    problem.ref_chi = problem.body_chi
    problem.imp_mask = imp_mask
    problem.imp_size = imp_size
    write_field(problem.body_chi, 'chi', 'chi.xmf', domain)
    return problem, state


def main():
    global frame, prevreport, history, problem, args

    args = parse_args()

    # Replace with relative path to output directory
    if args.ref_path is not None:
        args.ref_path = os.path.relpath(args.ref_path, start=args.outdir)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Change current directory to output directory.
    os.chdir(outdir)
    odil.util.set_log_file(open("train.log", 'w'))
    printlog(' '.join(sys.argv))
    printlog("Entering '{}'".format(outdir))

    with open(os.path.join('args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    if args.history_every:
        history = odil.History(csvpath='train.csv', warmup=1)
    else:
        history = None
    prevreport = argparse.Namespace()
    prevreport.time = time.time()
    prevreport.epoch = 0

    # Update arguments.
    args.mu = 2 * args.body_r / args.Re
    args.nlvl = min(args.nlvl, int(round(np.log2(args.N - 1))))
    args.plot_every *= args.every_factor
    args.history_every *= args.every_factor
    args.report_every *= args.every_factor
    args.checkpoint_every *= args.every_factor
    if args.epochs is None:
        args.epochs = args.frames * args.plot_every

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    problem, state = make_problem(args)
    frame = 0

    odil.util.optimize(args, args.optimizer, problem, state, callback)

    with open('done', 'w') as f:
        pass


if __name__ == "__main__":
    main()
