import numpy as np
import scipy.sparse
import scipy.sparse.linalg

try:
    import pyamg
except ModuleNotFoundError:
    pyamg = None

try:
    import cupy
    import cupyx.scipy.sparse
    import cupyx.scipy.sparse.linalg
except ModuleNotFoundError:
    cupy = None
    cupyx = None


def get_sparse_eye(size):
    return scipy.sparse.diags(np.ones(size), format='csr')


def solve(matr, rhs, args, history=None, linsolver="direct"):
    if args.linsolver_maxiter is None:
        if args.linsolver == 'lsqr':
            args.linsolver_maxiter = 1000
        else:
            args.linsolver_maxiter = 50

    eye = get_sparse_eye(matr.shape[1])
    matr_reg = matr.T.dot(matr).tocsr()
    if args.beta:
        matr_reg += args.beta**2 * eye
    if args.betadiag:
        matr_reg += args.betadiag**2 * scipy.sparse.diags(matr_reg.diagonal())
    rhs_reg = matr.T.dot(rhs)
    if linsolver == "direct":
        sol = scipy.sparse.linalg.spsolve(matr_reg, rhs_reg)
    elif linsolver == "direct_cu":
        if cupy is None:
            raise ModuleNotFoundError(
                "Module CuPy not found. Install with 'pip install cupy-cuda110'"
            )
        matr_reg = cupyx.scipy.sparse.csr_matrix(matr_reg)
        rhs_reg = cupy.array(rhs_reg)
        sol = cupyx.scipy.sparse.linalg.spsolve(matr_reg, rhs_reg)
        sol = sol.get()
    elif linsolver == "lsqr":
        sol, _, itn, _, _, anorm, acond, arnorm = \
                scipy.sparse.linalg.lsqr(
            matr,
            rhs,
            damp=args.beta,
            atol=args.linsolver_tol,
            btol=args.linsolver_tol,
            iter_lim=args.linsolver_maxiter)[:8]
        history['linsolver_residual'].append(arnorm)
        history['linsolver_anorm'].append(anorm)
        history['linsolver_acond'].append(acond)
        history['linsolver_niter'].append(itn)
    elif linsolver == "lsqr_cu":
        if cupy is None:
            raise ModuleNotFoundError(
                "Module CuPy not found. Install with 'pip install cupy-cuda110'"
            )
        # XXX cupy does not support non-square matrices
        sol = cupyx.scipy.sparse.linalg.lsqr(matr, rhs)[0]
    elif linsolver == "multigrid":
        if pyamg is None:
            raise ModuleNotFoundError(
                "Module PyAMG not found. Install with 'pip install pyamg'")
        ml = pyamg.smoothed_aggregation_solver(matr_reg)
        residuals = []
        sol = ml.solve(b=rhs_reg,
                       tol=args.linsolver_tol,
                       residuals=residuals,
                       accel='cg',
                       maxiter=args.linsolver_maxiter)
        history['linsolver_residual'].append(residuals[-1])
        history['linsolver_niter'].append(len(residuals))
    else:
        raise ValueError("Unknown linsolver=" + linsolver)

    return sol


def add_arguments(parser):
    parser.add_argument(
        '--linsolver',
        type=str,
        choices=["multigrid", "direct", "direct_cu", "lsqr", "lsqr_cu"],
        default="direct",
        help="Linear solver to use")
    parser.add_argument('--linsolver_maxiter',
                        type=int,
                        default=None,
                        help="Maximum number of iterations of linear solver")
    parser.add_argument('--linsolver_tol',
                        type=float,
                        default=1e-6,
                        help="Tolerance for linear solver")
    parser.add_argument('--beta',
                        type=float,
                        default=0,
                        help="Relaxation factor (0: no relaxation)")
    parser.add_argument('--betadiag',
                        type=float,
                        default=0,
                        help="Multiplier for diagonal (0: no relaxation)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
