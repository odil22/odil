#!/usr/bin/env python3

import numpy as np
from collections import defaultdict
import scipy.sparse as sp
import sys

g_verbose = False
g_reuse_multigridop = True


def get_shift_csr(dw, nw, zeropad=True, mod=None, dtype=None):
    '''
    Returns a shift matrix on a grid of size `nw`.
    Acting on a field, the matrix gives an element at offset `dw`.
    The operation is equivalent to a circulat shift of the field by `-dw`.
    If `zeropad` is True, the effect is equivalent to a noncircular shift.
    '''
    dim = len(nw)
    n = np.prod(nw)
    row = mod.reshape(mod.arange(n), nw)
    col = mod.roll(row, np.negative(list(dw)), axis=range(dim))
    row = mod.reshape(row, [-1])
    col = mod.reshape(col, [-1])
    if zeropad:
        data = mod.spnative(mod.zeros(nw, dtype))
        sel = tuple(
            slice(0, -s) if s > 0 else  #
            slice(-s, None) if s < 0 else slice(None) for s in dw)
        data[sel] = 1
        data = data.flatten()
    else:
        data = mod.ones(n)
    row = mod.spnative(row)
    col = mod.spnative(col)
    data = mod.spnative(data)
    res = mod.csr_matrix((data, (row, col)), shape=(n, n))
    return res


def noncircular_shift_np(u, shift, mod=None):
    '''
    Performs a non-circular shift of an array.
    Analogous to `np.roll()` but the vacant elements are filled with zeros.
    '''
    res = mod.zeros_like(u)
    iu = tuple(
        slice(0, -s) if s > 0 else slice(-s, None) if s < 0 else slice(None)
        for s in shift)
    ires = tuple(
        slice(s, None) if s > 0 else slice(0, s) if s < 0 else slice(None)
        for s in shift)
    res[ires] = u[iu]
    return res


def noncircular_shift(u, shift, mod=None):
    '''
    Performs a non-circular shift of an array.
    Analogous to `np.roll()` but the vacant elements are filled with zeros.
    '''
    paddings = [[max(0, s), max(0, -s)] for s in shift]
    iu = tuple(
        slice(0, -s) if s > 0 else slice(-s, None) if s < 0 else slice(None)
        for s in shift)
    res = mod.pad(u, paddings)[iu]
    return res


class Multigrid:

    def __init__(self,
                 nw,
                 nvar=1,
                 nlevels=None,
                 restriction='full',
                 mod=None,
                 dtype=None):
        '''
        nw: `tuple`
            Base grid size.
        nvar: `int`
            Number of unknown fields.
        '''
        nw = np.array(nw)
        max_nlevels = 0
        while 2**(max_nlevels + 1) < min(nw):
            max_nlevels += 1
        nlevels = max_nlevels if nlevels is None else min(nlevels, max_nlevels)
        # Grid size for each level.
        nnw = [nw]
        for i in range(1, nlevels):
            nnw.append((nnw[i - 1] - 1) // 2 + 1)

        # Interpolation matrices.
        TT = [self.get_T(nw, mod=mod, dtype=dtype) for nw in nnw]
        TT = [mod.block_diag([T] * nvar) for T in TT]
        # Restriction matrices.
        RRsingle = [
            self.get_R(nw, restriction=restriction, mod=mod, dtype=dtype)
            for nw in nnw
        ]
        RR = [mod.block_diag([R] * nvar) for R in RRsingle]

        self.dim = len(nw)
        self.nvar = nvar
        self.nlevels = nlevels
        self.nnw = nnw
        self.TT = TT
        self.RR = RR
        self.RRsingle = RRsingle
        self.mod = mod
        self.dtype = dtype
        self.restriction = restriction

    def update_A(self, A0, DD=None, compute_lower=False):
        # Discretization matrices.
        if isinstance(A0, list):
            assert len(A0) == self.nlevels, (
                "Expected len(A0)={:}, got {:}".format(self.nlevels, len(A0)))
            AA = A0
        else:
            AA = [None] * self.nlevels
            AA[0] = A0
            for level in range(1, self.nlevels):
                if g_verbose:
                    sys.stderr.write('{:} '.format(level))
                    sys.stderr.flush()
                R = self.RR[level - 1]
                T = self.TT[level - 1]
                AA[level] = R @ AA[level - 1] @ T

        def extdiag(matr, nvar):
            res = []
            ns = matr.shape[0] // nvar
            res = [[
                matr[i * ns:(i + 1) * ns, j * ns:(j + 1) * ns].diagonal(0)
                for j in range(nvar)
            ] for i in range(nvar)]
            return res

        if g_verbose:
            sys.stderr.write('D ')
            sys.stderr.flush()
        # Diagonal parts of AA.
        DD = [extdiag(A, self.nvar) for A in AA]
        # Lower triangular parts of AA, including diagonals.
        if compute_lower:
            LL = [mod.tril(A).tocsr() for A in AA]
        else:
            LL = [None] * self.nlevels
        self.AA = AA
        self.DD = DD
        self.LL = LL
        if g_verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

    @staticmethod
    def geom(nw):
        # Space dimensionality.
        dim = len(nw)

        # Size of coarser grid.
        nwh = nw // 2 + 1

        # Indices of points of current grid.
        iiw = np.meshgrid(*[range(n) for n in nw], indexing='ij')
        iiw = np.reshape(iiw, (dim, -1)).T

        # Indices of points of coarser grid.
        iiwh = np.meshgrid(*[range(n) for n in nwh], indexing='ij')
        iiwh = np.reshape(iiwh, (dim, -1)).T

        # Multipliers to convert multi-index to flat index.
        ow = np.array([1] * dim)
        for i in range(0, dim - 1):
            ow[i + 1] = ow[i] * nw[i]
        owh = np.array([1] * dim)
        for i in range(0, dim - 1):
            owh[i + 1] = owh[i] * nwh[i]

        def flat(iw, ow):
            '''
            Returns flat index from multi-index `iw` using multipliers `ow`.
            '''
            return np.sum(iw * ow)

        return dim, nwh, ow, owh, iiw, iiwh, flat

    @staticmethod
    def get_T(nw, mod=None, dtype=None):
        '''
        Returns interpolation matrix to `n` points, shape (n, n // 2 + 1).
        '''
        if g_reuse_multigridop:
            return MultigridOp.get_TT(nw, mod=mod, dtype=dtype).tocsr().T
        nw = np.array(nw)
        row = []
        col = []
        data = []

        dim, nwh, ow, owh, iiw, iiwh, flat = Multigrid.geom(nw)

        def append(irow, iwh, c):
            row.append(irow)
            col.append(flat(iwh, owh))
            data.append(c)

        ddw = np.meshgrid(*[[0, 1]] * dim, indexing='ij')
        ddw = np.reshape(ddw, (dim, -1)).T
        ddw = [StencilDict.multi_index(dw) for dw in ddw]

        for iw in iiw:
            irow = flat(iw, ow)
            iwh = iw // 2
            rw = StencilDict.multi_index(iw % 2)
            for dw in ddw:
                if all(dw <= rw):
                    append(irow, iwh + dw, 0.5**sum(rw))
        res = mod.csr_matrix((data, (row, col)),
                             shape=(np.prod(nw), np.prod(nwh)))
        return res

    @staticmethod
    def get_R(nw, restriction='full', mod=None, dtype=None):
        '''
        Returns restriction matrix from `n` points, shape (n // 2 + 1, n).
        '''
        if g_reuse_multigridop:
            return MultigridOp.get_R(nw,
                                     mod=mod,
                                     restriction=restriction,
                                     dtype=dtype).tocsr()
        nw = np.array(nw)
        row = []
        col = []
        data = []

        dim, nwh, ow, owh, iiw, iiwh, flat = Multigrid.geom(nw)

        def append(irow, iw, c):
            row.append(irow)
            col.append(flat(iw, ow))
            data.append(c)

        if restriction == 'full':
            coeff = lambda dw: 0.5**sum(abs(dw))
        elif restriction == 'half':
            coeff = lambda dw: (2 * dim if sum(abs(dw)) == 0 else 1
                                if sum(abs(dw)) == 1 else 0)
        elif restriction == 'injection':
            coeff = lambda dw: 1 if sum(abs(dw)) == 0 else 0
        else:
            raise ValueError("Unknown restriction=" + restriction)

        # Stencil indices.
        ddw = np.meshgrid(*[[-1, 0, 1]] * dim, indexing='ij')
        ddw = np.reshape(ddw, (dim, -1)).T
        ddw = [StencilDict.multi_index(dw) for dw in ddw]

        # Sum of coefficients.
        csum = 0
        for dw in ddw:
            csum += coeff(dw)

        for iwh in iiwh:
            irow = flat(iwh, owh)
            iw = iwh * 2
            if np.all(iwh > 0) and np.all(iwh + 1 < nwh):
                # Inner node.
                for dw in ddw:
                    dw = np.array(list(dw))
                    append(irow, iw + dw, coeff(dw) / csum)
            else:
                # Boundary node.
                append(irow, iw, 1.)

        res = sp.csr_matrix((data, (row, col)),
                            shape=(np.prod(nwh), np.prod(nw)))
        res.eliminate_zeros()
        return res

    def smoother_jacobi(self, level, u, f, omega=1., full=False):
        A = self.AA[level]
        D = self.DD[level]
        nvar = self.nvar
        r = f - A @ u
        if full:  # Full inversion using diagonals from all blocks.
            mod = self.mod
            du = mod.zeros_like(f)
            d = mod.moveaxis(np.array(D), 2, 0)  # Shape (n, nvar, nvar).
            r = mod.reshape(r, (self.nvar, -1)).T  # Shape (n, nvar).
            du = mod.solve(d, r).T.flatten()
        else:  # Only use diagonals from diagonal blocks.
            d = np.hstack([D[i][i] for i in range(nvar)])
            du = r / d
        u += du * omega
        return u

    def smoother_gauss_seidel(self, level, u, f, omega=1.):
        A = self.AA[level]
        L = self.LL[level]
        u += (sp.linalg.spsolve_triangular(L, f - (A @ u - L @ u)) - u) * omega
        return u

    def smoother_direct(self, level, u, f):
        A = self.AA[level]
        u = self.mod.spsolve(A, f)
        return u

    def step(self, u, f, smoother, level=0, ndirect=5, pre=2, post=2):
        '''
        smoother: `callable`
            Function `smoother(level, u, f)` returning an approximate solution of
               AA[level] u = f
            using initial guess `u`.
        '''
        if g_verbose:
            sys.stderr.write('{:} '.format(level))
            sys.stderr.flush()
        if level + 1 >= self.nlevels or min(self.nnw[level]) <= ndirect:
            return self.smoother_direct(level, u, f)
        # Pre-smoothing.
        for _ in range(pre):
            u = smoother(level, u, f)
        A = self.AA[level]
        R = self.RR[level]
        T = self.TT[level]
        r = R @ (f - A @ u)
        u = u + T @ self.step(
            np.zeros_like(r), r, smoother, level=level + 1, ndirect=ndirect)
        # Post-smoothing.
        for _ in range(post):
            u = smoother(level, u, f)
        if g_verbose and level == 0:
            sys.stderr.write('\n')
            sys.stderr.flush()
        return u


class MultigridOp:
    '''
    Implementation of Multigrid representing matrices using SparseOperator.
    '''

    def __init__(self,
                 nw,
                 nvar=1,
                 nlevels=None,
                 mod=None,
                 dtype=None,
                 restriction='full'):
        '''
        nw: `tuple`
            Base grid size.
        nvar: `int`
            Number of unknown fields.
        '''
        assert mod is not None
        assert dtype is not None
        nw = np.array(nw)
        max_nlevels = 0
        while 2**(max_nlevels + 1) < min(nw):
            max_nlevels += 1
        nlevels = max_nlevels if nlevels is None else min(nlevels, max_nlevels)
        # Grid size for each level.
        nnw = [nw]
        for i in range(1, nlevels):
            nnw.append((nnw[i - 1] - 1) // 2 + 1)

        # Interpolation matrices.
        TT = [self.get_TT(nw, mod=mod, dtype=dtype) for nw in nnw]
        # Restriction matrices.
        RR = [
            self.get_R(nw, mod=mod, dtype=dtype, restriction=restriction)
            for nw in nnw
        ]

        self.mod = mod
        self.nvar = nvar
        self.dtype = dtype
        self.dim = len(nw)
        self.nlevels = nlevels
        self.nnw = nnw
        self.TT = TT
        self.RR = RR
        self.restriction = restriction

    @staticmethod
    def coarsen_sparse_operator(A, R, TT, mod=None, dtype=None):
        '''
        A: `SparseOperator`
            Operator on grid with `nw` points.
        R: `SparseOperator`
            Restriction operator from `nw` points to `nwh` points.
            Can be generated using `Multigrid.get_R_op(nw)`.
        TT: `SparseOperator`
            Transpose of interpolation operator from `nwh` points to `nw` points.
            Can be generated using `Multigrid.get_TT_op(nw)`.

        Returns:
            Ah: Operator `A` restricted to grid with `nwh` points.
        '''

        nw = R.nw
        nwh = R.nwh
        dim = len(nw)
        ddwc = np.meshgrid(*[[-2, -1, 0, 1, 2]] * dim, indexing='ij')
        ddwc = np.reshape(ddwc, (dim, -1)).T
        ddwc = [StencilDict.multi_index(dwc) for dwc in ddwc]
        res = StencilDict([], [], mod=mod)
        for dwc in ddwc:
            mc = mod.zeros(nwh.prod(), dtype=dtype)
            for dwa, ma in A.shift_to_field.items():
                if mod.sum(ma**2) == 0:
                    continue
                if not all(abs(dwa - dwc * 2) <= 2):
                    continue
                M = R.mul_elementwise(TT.shift_left(-dwc).shift_right(dwa))
                mc += M.mul_field(ma)
            if mod.sum(mc**2) == 0:
                continue
            res.append(dwc, mc)
        res.merge_duplicates()
        Ah = SparseOperator(res, nwh, mod=mod, dtype=dtype)
        return Ah

    def update_A(self, A0, set_lower=False):
        assert isinstance(A0, list)
        assert isinstance(A0[0], list)
        nvar = self.nvar
        mod = self.mod
        dtype = self.dtype
        # Discretization matrices.
        AA = [None] * self.nlevels
        AA[0] = A0
        for level in range(1, self.nlevels):
            if g_verbose:
                sys.stderr.write('{:} '.format(level))
                sys.stderr.flush()
            R = self.RR[level - 1]
            T = self.TT[level - 1]
            AA[level] = [[
                self.coarsen_sparse_operator(
                    AA[level - 1][i][j],
                    R,
                    T,
                    mod=mod,
                    dtype=dtype,
                ) for j in range(nvar)
            ] for i in range(nvar)]
        if g_verbose:
            sys.stderr.write('D ')
            sys.stderr.flush()
        # Diagonal parts of AA.
        dwz = StencilDict.multi_index((0, ) * self.dim)
        DD = [[
            mod.reshape(
                A[i][i].shift_to_field.get(dwz,
                                           mod.zeros(A[i][i].nw, dtype=dtype)),
                [-1]) for i in range(nvar)
        ] for A in AA]
        self.AA = AA
        self.DD = DD

        if g_verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

    @staticmethod
    def get_TT(nw, mod=None, dtype=None):
        '''
        Returns transpose of interpolation operator
        to grid of size `nw` from coarser grid.
        '''
        nw = np.array(nw)
        dim = len(nw)
        nwh = nw // 2 + 1

        # Stencil indices.
        ddw = np.meshgrid(*[[-1, 0, 1]] * dim, indexing='ij')
        ddw = np.reshape(ddw, (dim, -1)).T
        ddw = [StencilDict.multi_index(dw) for dw in ddw]

        res = StencilDict([], [], mod)
        for dw in ddw:
            # Initialize with inner values.
            u = mod.full(nwh - abs(dw), mod.cast(0.5**(sum(abs(dw))), dtype))
            # Pad with zeros.
            paddings = [[max(0, -s), max(0, s)] for s in dw]
            res.append(dw, mod.pad(u, paddings))

        return SparseOperator(res, nw, stride=2, mod=mod, dtype=dtype)

    @staticmethod
    def get_R(nw, mod=None, dtype=None, restriction='full'):
        '''
        Returns restriction operator from grid of size `nw` to coarser grid.
        '''
        nw = np.array(nw)
        dim = len(nw)
        nwh = nw // 2 + 1

        # Stencil indices.
        if restriction == 'full':
            ddw = np.meshgrid(*[[-1, 0, 1]] * dim, indexing='ij')
            ddw = np.reshape(ddw, (dim, -1)).T
            coeff = lambda dw: 0.5**(sum(abs(dw)) + dim)
        elif restriction == 'half':
            ddw = [(0, ) * dim]
            for i in range(dim):
                ddw.append(tuple(1 if j == i else 0 for j in range(dim)))
                ddw.append(tuple(-1 if j == i else 0 for j in range(dim)))
            coeff = lambda dw: (0.5 if sum(abs(dw)) == 0 else 0.25 / dim
                                if sum(abs(dw)) == 1 else 0)
        elif restriction == 'injection':
            ddw = [(0, ) * dim]
            coeff = lambda dw: 1 if sum(abs(dw)) == 0 else 0
        else:
            raise ValueError("Unknown restriction=" + restriction)

        ddw = [StencilDict.multi_index(dw) for dw in ddw]

        res = StencilDict([], [], mod)
        dwz = StencilDict.multi_index((0, ) * dim)
        for dw in ddw:
            # Initialize with inner values.
            u = mod.full(nwh - 2, mod.cast(coeff(dw), dtype))
            paddings = [[1, 1]] * dim
            val = mod.numpy(mod.cast(1 if all(dw == dwz) else 0, dtype))
            res.append(dw, mod.pad(u, paddings, constant_values=val))

        return SparseOperator(res, nw, stride=2, mod=mod, dtype=dtype)

    def residual(self, A, u, f):
        nvar = self.nvar
        r = [None] * nvar
        for i in range(nvar):
            r[i] = f[i] - sum(A[i][j].mul_field(u[j]) for j in range(nvar))
        return r

    def smoother_jacobi(self, level, u, f, omega=1.):
        assert isinstance(u, list)
        assert isinstance(f, list)
        A = self.AA[level]
        D = self.DD[level]
        nvar = self.nvar
        r = self.residual(A, u, f)
        for i in range(nvar):
            u[i] += r[i] / D[i] * omega
        return u

    def smoother_direct(self, level, u, f):
        assert isinstance(u, list)
        assert isinstance(f, list)
        mod = self.mod
        nvar = self.nvar
        f = mod.reshape(mod.stack(f), [-1])
        A = self.AA[level]
        A = [[A[i][j].tocsr() for j in range(nvar)] for i in range(nvar)]
        A = mod.bmat(A)
        u = mod.spsolve(A, f)
        u = list(mod.reshape(u, (nvar, -1)))
        return u

    def step(self, u, f, smoother, level=0, ndirect=5, pre=2, post=2):
        '''
        smoother: `callable`
            Function `smoother(level, u, f)` returning an approximate solution of
               AA[level] u = f
            using initial guess `u`.
        '''
        assert isinstance(u, list)
        assert isinstance(f, list)
        if g_verbose:
            sys.stderr.write('{:} '.format(level))
            sys.stderr.flush()
        mod = self.mod
        dtype = self.dtype
        if level + 1 >= self.nlevels or min(self.nnw[level]) <= ndirect:
            return self.smoother_direct(level, u, f)
        # Pre-smoothing.
        for _ in range(pre):
            u = smoother(level, u, f)
        A = self.AA[level]
        R = self.RR[level]
        T = self.TT[level]
        r = [R.mul_field(r) for r in self.residual(A, u, f)]
        du = self.step([mod.zeros_like(r, dtype=dtype) for r in r],
                       r,
                       smoother,
                       level=level + 1)
        du = [T.mul_transpose_field(du) for du in du]
        u = [u + du for u, du in zip(u, du)]
        # Post-smoothing.
        for _ in range(post):
            u = smoother(level, u, f)
        if g_verbose and level == 0:
            sys.stderr.write('\n')
            sys.stderr.flush()
        return u


class MultiIndex:

    def __init__(self, *indices, bits=2):
        self.indices = [int(i) for i in indices]
        self.bits = bits

    def __add__(self, other):
        return MultiIndex(
            *[i + j for i, j in zip(self.indices, other.indices)])

    def __sub__(self, other):
        return MultiIndex(
            *[i - j for i, j in zip(self.indices, other.indices)])

    def __neg__(self):
        return MultiIndex(*[-i for i in self.indices])

    def __mod__(self, div):
        return MultiIndex(*[i % div for i in self.indices])

    def __mul__(self, mul):
        return MultiIndex(*[i * mul for i in self.indices])

    def __abs__(self):
        return MultiIndex(*[abs(i) for i in self.indices])

    def serialize(self):
        res = int(
            sum((i + 2**self.bits) << ((1 + self.bits) * n)
                for n, i in enumerate(self.indices)))
        return res

    def __hash__(self):
        return self.serialize()

    def __str__(self):
        return 'MultiIndex({})'.format(', '.join(map(str, self.indices)))

    def __iter__(self):
        return iter(self.indices)

    def __le__(self, other):
        if isinstance(other, int):
            return all([i <= other for i in self.indices])
        return all([i <= j for i, j in zip(self.indices, other.indices)])

    def __eq__(self, other):
        if isinstance(other, int):
            return all([i == other for i in self.indices])
        return all([i == j for i, j in zip(self.indices, other.indices)])

    def __sum__(self):
        return sum(self.indices)

    def minimum(self, other):
        if isinstance(other, int):
            return [min(i, other) for i in self.indices]
        return [min(i, j) for i, j in zip(self.indices, other.indices)]


class StencilDict:

    def __init__(self, keys, values, mod=None):
        assert len(keys) == len(values)
        assert mod is not None
        self._keys = keys
        self._values = values
        self.mod = mod

    def items(self):
        return zip(self._keys, self._values)

    def keys(self):
        return self._keys

    def values(self):
        return self._values

    @staticmethod
    def serialize(dw, bits=3):
        res = int(
            sum((i + 2**bits) << ((1 + bits) * n) for n, i in enumerate(dw)))
        return res

    @staticmethod
    def multi_index(dw):
        try:
            return np.array(dw)
        except NotImplementedError:
            return dw

    @staticmethod
    def minimum(dw, other):
        return np.minimum(dw, other)

    def merge_duplicates(self):
        return
        mod = self.mod

        indices = []
        for i, dw in enumerate(self.shift_to_field.keys()):
            indices.append((dw, StencilDict.serialize(dw), i))
        # Sort by serialized.
        indices = sorted(indices, key=lambda x: x[1])
        same = []  # Lists of elments with the same index.
        sprev = None
        for dw, s, i in indices:
            if sprev != s:
                same.append([])
            same[-1].append(dw, i)

        keys = []
        values = []
        for series in same:
            c = mod.zeros(self.nw, self.dtype)
            for dw, i in series:
                c += self._values[i]
            keys.append(dw)
            values.append(c)

        self._keys = keys
        self._values = values

    def append(self, key, value):
        self._keys.append(key)
        self._values.append(value)

    def lookup(self, key):
        if key in self._keys:
            return self._vales[self._keys.index(key)]
        return None

    def __len__(self):
        return len(self._keys)

    def get(self, key, default):
        for k, v in self.items():
            if all(k == key):
                return v
        return default


class ModBase:

    def __init__(self, mod=None):
        assert mod is not None
        self.mod = mod
        for name in [
                'cos',
                'float32',
                'float64',
                'linspace',
                'ones',
                'ones_like',
                'roll',
                'pad',
                'reshape',
                'stack',
                'sin',
                'zeros',
                'zeros_like',
        ]:
            setattr(self, name, getattr(mod, name))
        self.flatten = lambda x: mod.reshape(x, [-1])


class ModNumpy(ModBase):

    def __init__(self, mod=None, modsp=None):
        super().__init__(mod)
        self.cast = lambda x, t: mod.array(x, dtype=t)
        self.numpy = mod.array
        self.native = mod.array
        self.spnative = lambda x: x
        self.full = mod.full
        self.sum = mod.sum
        self.arange = mod.arange
        self.norm = mod.linalg.norm
        self.solve = mod.linalg.solve
        self.mod = mod
        self.modsp = modsp
        self.moveaxis = mod.moveaxis
        self.hstack = mod.hstack
        if modsp is not None:
            self.csr_matrix = modsp.csr_matrix
            self.diags = modsp.diags
            self.bmat = modsp.bmat
            self.block_diag = modsp.block_diag
            self.tril = modsp.tril
            tmp = self.numpy
            self.numpy = (lambda x: x
                          if isinstance(x, modsp.csr_matrix) else tmp(x))
            self.spnorm = modsp.linalg.norm
            self.spsolve = modsp.linalg.spsolve


class ModTensorflow(ModBase):

    def __init__(self, mod=None, modsp=None):
        super().__init__(mod)
        self.batch_to_space = mod.batch_to_space
        self.cast = mod.cast
        self.numpy = lambda x: x.numpy()
        self.native = lambda x: x
        self.spnative = lambda x: x.numpy() if hasattr(x, 'numpy') else x
        self.sum = mod.reduce_sum
        self.full = mod.fill
        self.arange = mod.range
        self.norm = lambda x: mod.reduce_sum(x**2)**0.5
        self.mod = mod
        self.modsp = modsp
        if modsp is not None:

            def csr_matrix(*args, **kwargs):
                if 'dtype' in kwargs:
                    kwargs['dtype'] = {
                        mod.float32: np.float32,
                        mod.float64: np.float64
                    }[kwargs['dtype']]
                return modsp.csr_matrix(*args, **kwargs)

            self.csr_matrix = csr_matrix
            self.diags = modsp.diags
            self.bmat = modsp.bmat
            self.block_diag = modsp.block_diag
            self.tril = modsp.tril
            tmp = self.numpy
            self.numpy = (lambda x: x
                          if isinstance(x, modsp.csr_matrix) else tmp(x))
            self.spnorm = modsp.linalg.norm
            self.spsolve = modsp.linalg.spsolve


class ModCupy(ModBase):

    def __init__(self, mod=None, modsp=None):
        super().__init__(mod)
        self.cast = lambda x, t: mod.array(x, dtype=t)
        self.numpy = lambda x: x.get() if isinstance(x, mod.ndarray) else x
        self.full = mod.full
        self.sum = mod.sum
        self.native = mod.array
        self.spnative = lambda x: x
        self.arange = mod.arange
        self.norm = mod.linalg.norm
        self.solve = mod.linalg.solve
        self.mod = mod
        self.modsp = modsp
        self.moveaxis = mod.moveaxis
        self.hstack = mod.hstack
        if modsp is not None:
            self.csr_matrix = modsp.csr_matrix
            self.diags = modsp.diags
            self.bmat = modsp.bmat

            def block_diag(bb):
                n = len(bb)
                res = [[None for _ in range(n)] for _ in range(n)]
                for i in range(n):
                    for j in range(n):
                        res[i][i] = bb[i]
                return modsp.bmat(res, format='csr')

            self.block_diag = block_diag
            self.tril = modsp.tril
            tmp = self.numpy
            self.numpy = (lambda x: x.get()
                          if isinstance(x, modsp.csr_matrix) else tmp(x))
            self.spnorm = modsp.linalg.norm
            self.spsolve = modsp.linalg.spsolve


class SparseOperator:

    def __init__(self, shift_to_field, nw, stride=1, mod=None, dtype=None):
        '''
        shift_to_field: `dict`
            Mapping from a shift to the corresponding coefficient. The sparse
            operator is a sum of shift operators with the specified
            coefficients.
        nw: `tuple`
            Grid size.
        stride: `int`
            Stride of output, this will restrict the output to a coarser grid.
        '''
        assert mod is not None
        assert dtype is not None
        self.mod = mod
        self.modnp = ModNumpy(np, sp)
        self.dtype = dtype
        self.nw = np.array(nw)
        self.nwh = np.array((self.nw - 1) // stride + 1)
        self.shift_to_field = StencilDict(
            [StencilDict.multi_index(dw) for dw in shift_to_field.keys()],
            [mod.reshape(c, self.nwh) for c in shift_to_field.values()],
            mod=mod)
        self.dim = len(nw)
        self.stride = stride
        for c in self.shift_to_field.values():
            assert np.prod(c.shape) == self.nwh.prod()

    def tocsr(self):
        mod = self.mod
        n = self.nw.prod()
        nh = self.nwh.prod()
        res = self.mod.csr_matrix((nh, n), dtype=self.dtype)
        # Indices to apply stride.
        istride = mod.reshape(mod.arange(n), self.nw)
        sel = (slice(None, None, self.stride), ) * self.dim
        istride = mod.reshape(istride[sel], [-1])
        for dw, c in self.shift_to_field.items():
            c = mod.spnative(c).flatten()
            m = get_shift_csr(dw, self.nw, mod=mod, dtype=self.dtype)[istride]
            res += mod.diags([c], [0], format='csr') @ m
        res.eliminate_zeros()
        return res

    @staticmethod
    def eye(nw, stride=1, mod=None, dtype=None):
        dw = (0, ) * len(nw)
        res = StencilDict([dw], [mod.ones(nw)], mod=mod)
        return SparseOperator(res, nw, stride=stride, mod=mod, dtype=dtype)

    @staticmethod
    def diag(c, nw, stride=1, mod=None):
        dw = (0, ) * len(nw)
        res = StencilDict([dw], [c], mod=mod)
        return SparseOperator(res, nw, stride=stride, mod=mod, dtype=c.dtype)

    @staticmethod
    def apply_stride(u, stride, nw, mod=None):
        u = mod.reshape(u, nw)
        dim = len(nw)
        sel = (slice(None, None, stride), ) * dim
        return u[sel]

    @staticmethod
    def apply_shift(u, dw, nw, mod=None):
        '''
        Returns the result of `get_shift_csr(dw)` applied to field `u`.
        '''
        u = mod.reshape(u, nw)
        return noncircular_shift(u, -StencilDict.multi_index(dw), mod=mod)

    def eliminate_zeros(self):
        keys = []
        values = []
        res = StencilDict([], [], mod=self.mod)
        for dw, c in self.shift_to_field.items():
            if self.mod.sum(c**2):
                res.append(dw, c)
        self.shift_to_field = res

    def mul_field(self, u):
        mod = self.mod
        dtype = self.dtype
        u = mod.reshape(u, self.nw)
        res = mod.zeros(np.prod(self.nwh), dtype=dtype)
        for dw, c in self.shift_to_field.items():
            uh = self.apply_stride(self.apply_shift(u, dw, self.nw, mod=mod),
                                   self.stride,
                                   self.nw,
                                   mod=mod)
            res += mod.reshape(c, [-1]) * mod.reshape(uh, [-1])
        return mod.reshape(res, [-1])

    def mul_transpose_field_tf(self, uh):
        mod = self.mod
        dim = self.dim
        stride = self.stride
        assert stride in [1, 2], "Unsupported stride={:}".format(stride)

        uh = mod.reshape(uh, self.nwh)

        if stride == 2:
            ddwr = np.meshgrid(*[[0, 1]] * dim, indexing='ij')
            ddwr = np.reshape(ddwr, (dim, -1)).T
            ddwr = [StencilDict.multi_index(dwr) for dwr in ddwr]
            res = {
                StencilDict.serialize(dwr): mod.zeros(self.nwh,
                                                      dtype=self.dtype)
                for dwr in ddwr
            }
            for dw, c in self.shift_to_field.items():
                dwr = abs(dw) % 2
                cs = noncircular_shift(c, StencilDict.minimum(dw, 0), mod=mod)
                uhs = noncircular_shift(uh,
                                        StencilDict.minimum(dw, 0),
                                        mod=mod)
                res[StencilDict.serialize(dwr)] += cs * uhs
            res = mod.batch_to_space(list(res.values()),
                                     block_shape=[2] * dim,
                                     crops=[[0, 1]] * dim)[0]
        elif stride == 1:
            res = mod.zeros(self.nw, dtype=self.dtype)
            for dw, c in self.shift_to_field.items():
                cs = noncircular_shift(c, dw, mod=mod)
                uhs = noncircular_shift(uh, dw, mod=mod)
                res += cs * uhs
        return mod.reshape(res, [-1])

    def mul_transpose_field_np(self, uh):
        mod = self.mod
        stride = self.stride
        assert stride in [1, 2], "Unsupported stride={:}".format(stride)

        uh = mod.reshape(uh, self.nwh)
        res = mod.zeros(self.nw, dtype=self.dtype)

        if stride == 2:
            for dw, c in self.shift_to_field.items():
                sel = tuple(slice(abs(s), None, stride) for s in dw)
                iu = tuple(
                    slice(0, -s) if s > 0 else  #
                    slice(-s, None) if s < 0 else slice(None) for s in dw)
                res[sel] += c[iu] * uh[iu]
        elif stride == 1:
            res = mod.zeros(self.nw, dtype=self.dtype)
            for dw, c in self.shift_to_field.items():
                cs = noncircular_shift(c, dw, mod=mod)
                uhs = noncircular_shift(uh, dw, mod=mod)
                res += cs * uhs
        return mod.reshape(res, [-1])

    def mul_transpose_field(self, uh):
        if hasattr(self.mod, 'batch_to_space'):
            return self.mul_transpose_field_tf(uh)
        else:
            return self.mul_transpose_field_np(uh)

    def mul_transpose_op(self, op):
        assert self.stride == 1
        assert op.stride == 1
        mod = self.mod
        dtype = self.dtype
        pairs = []
        stf_a = self.shift_to_field
        stf_b = op.shift_to_field
        for ia, dwa in enumerate(stf_a.keys()):
            for ib, dwb in enumerate(stf_b.keys()):
                pairs.append(((dwa, ia), (dwb, ib)))
        pairs = sorted(pairs,
                       key=lambda x: StencilDict.serialize(x[1][0] - x[0][0]))
        samediff = []  # Lists of pairs with the same difference `dwb-dwa`.
        sprev = None
        for pa, pb in pairs:
            dwa = pa[0]
            dwb = pb[0]
            dwba = dwb - dwa
            if sprev != StencilDict.serialize(dwba):
                samediff.append([])
            samediff[-1].append((pa, pb))
            sprev = StencilDict.serialize(dwba)

        res = StencilDict([], [], mod=self.mod)
        for series in samediff:
            c = mod.zeros(self.nw, dtype)
            for pa, pb in series:
                dwa, ia = pa
                dwb, ib = pb
                ca = stf_a.values()[ia]
                cb = stf_b.values()[ib]
                c += self.apply_shift(ca * cb, -dwa, self.nw, mod=mod)
                dwba = dwb - dwa
            res.append(dwba, c)
        res = SparseOperator(res, self.nw, mod=mod, dtype=self.dtype)
        return res

    def mul_self_transpose(self):
        return self.mul_transpose_op(self)

    def mul_op(self, op):
        raise NotImplementedError
        assert np.all(self.nw == op.nwh), (
            "Dimensions do not match, expected equal {:} and {:}".format(
                self.nw, op.nwh))
        mod = self.mod
        dtype = self.dtype
        res = defaultdict(lambda: mod.zeros(op.nw, dtype))
        for dwa, ca in self.shift_to_field.items():
            for dwb, cb in op.shift_to_field.items():
                cb = noncircular_shift(cb, -dwa, mod=mod)
                cb = self.apply_stride(cb, self.stride, op.nw, mod=mod)
                res[dwa + dwb] += ca * cb
        return SparseOperator(res,
                              op.nw,
                              stride=self.stride * op.stride,
                              mod=mod,
                              dtype=dtype)

    def mul_elementwise(self, op):
        assert np.all(self.nw == op.nw), (
            "Dimensions do not match, expected equal {:} and {:}".format(
                self.nw, op.nw))
        assert self.stride == op.stride, (
            "Strides do not match, expected equal {:} and {:}".format(
                self.stride, op.stride))

        mod = self.mod
        stf_a = self.shift_to_field
        stf_b = op.shift_to_field
        series_a = [(k, StencilDict.serialize(k), i)
                    for i, k in enumerate(stf_a.keys())]
        series_b = [(k, StencilDict.serialize(k), i)
                    for i, k in enumerate(stf_b.keys())]
        # Sort by serialized.
        series_a = sorted(series_a, key=lambda x: x[1])
        series_b = sorted(series_b, key=lambda x: x[1])

        res = StencilDict([], [], mod=mod)
        ka = 0
        kb = 0
        while ka < len(series_a) and kb < len(series_b):
            dwa, sa, ia = series_a[ka]
            dwb, sb, ib = series_b[kb]
            assert ka == 0 or series_a[ka][1] != series_a[ka - 1][1]
            assert kb == 0 or series_b[kb][1] != series_b[kb - 1][1]
            if sa < sb:
                ka += 1
                continue
            if sb < sa:
                kb += 1
                continue
            assert sa == sb
            res.append(dwa, stf_a.values()[ia] * stf_b.values()[ib])
            ka += 1
            kb += 1
        return SparseOperator(res,
                              op.nw,
                              stride=self.stride,
                              mod=self.mod,
                              dtype=self.dtype)

    def add_elementwise(self, op):
        assert np.all(self.nw == op.nw), (
            "Dimensions do not match, expected equal {:} and {:}".format(
                self.nw, op.nw))
        assert self.stride == op.stride, (
            "Strides do not match, expected equal {:} and {:}".format(
                self.stride, op.stride))

        mod = self.mod
        stf_a = self.shift_to_field
        stf_b = op.shift_to_field
        series_a = [(k, StencilDict.serialize(k), i)
                    for i, k in enumerate(stf_a.keys())]
        series_b = [(k, StencilDict.serialize(k), i)
                    for i, k in enumerate(stf_b.keys())]
        # Sort by serialized.
        series_a = sorted(series_a, key=lambda x: x[1])
        series_b = sorted(series_b, key=lambda x: x[1])

        res = StencilDict([], [], mod=mod)
        ka = 0
        kb = 0
        none = (1 << 30)  # Number larger than all results of serialize().
        while ka < len(series_a) or kb < len(series_b):
            sa = series_a[ka][1] if ka < len(series_a) else none
            sb = series_b[kb][1] if kb < len(series_b) else none
            if sa < sb:
                dwa, _, ia = series_a[ka]
                res.append(dwa, stf_a.values()[ia])
                ka += 1
                continue
            if sb < sa:
                dwb, _, ib = series_b[kb]
                res.append(dwb, stf_b.values()[ib])
                kb += 1
                continue
            assert sa == sb
            dwa, _, ia = series_a[ka]
            dwb, _, ib = series_b[kb]
            res.append(dwa, stf_a.values()[ia] + stf_b.values()[ib])
            ka += 1
            kb += 1
        return SparseOperator(res,
                              op.nw,
                              stride=self.stride,
                              mod=self.mod,
                              dtype=self.dtype)

    def shift_left(self, shift):
        shift = StencilDict.multi_index(shift)
        s = shift * self.stride
        res = StencilDict(
            [dw + s for dw in self.shift_to_field.keys()],
            [
                self.apply_shift(c, shift, self.nwh, mod=self.mod)
                for c in self.shift_to_field.values()
            ],
            mod=self.mod,
        )
        return SparseOperator(res,
                              self.nw,
                              self.stride,
                              mod=self.mod,
                              dtype=self.dtype)

    def shift_right(self, shift):
        shift = StencilDict.multi_index(shift)
        res = StencilDict(
            [dw + shift for dw in self.shift_to_field.keys()],
            [c for c in self.shift_to_field.values()],
            mod=self.mod,
        )
        return SparseOperator(res,
                              self.nw,
                              self.stride,
                              mod=self.mod,
                              dtype=self.dtype)


class MultigridDecomp:

    @staticmethod
    def interp_field(u, rep=1, mod=None):
        dim = len(u.shape)
        if rep == 0:
            return u

        def term(*dd):
            dd = [tuple(-v for v in d) for d in dd]
            return sum(mod.roll(u, d, range(dim)) for d in dd) / len(dd)

        if dim == 3:
            u000 = u
            u100 = term((0, 0, 0), (1, 0, 0))
            u010 = term((0, 0, 0), (0, 1, 0))
            u001 = term((0, 0, 0), (0, 0, 1))
            u110 = term((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0))
            u011 = term((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1))
            u101 = term((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1))
            u111 = term((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), \
                        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
            uu = [u000, u001, u010, u011, u100, u101, u110, u111]
            u = mod.batch_to_space(uu,
                                   block_shape=[2] * dim,
                                   crops=[[0, 1]] * dim)[0]
        elif dim == 2:
            u00 = u
            u01 = term((0, 0), (0, 1))
            u10 = term((0, 0), (1, 0))
            u11 = term((0, 0), (0, 1), (1, 0), (1, 1))
            uu = [u00, u01, u10, u11]
            u = mod.batch_to_space(uu,
                                   block_shape=[2] * dim,
                                   crops=[[0, 1]] * dim)[0]
        elif dim == 1:
            u0 = u
            u1 = term((0, ), (1, ))
            uu = [u0, u1]
            u = mod.batch_to_space(uu,
                                   block_shape=[2] * dim,
                                   crops=[[0, 1]] * dim)[0]
        else:
            raise NotImplemented('Expected dim=1,2,3, got ' + int(dim))
        return MultigridDecomp.interp_field(u, rep - 1, mod=mod)

    @staticmethod
    def weights_to_fields(uw, nnw, mod):
        uu = []
        s = 0
        for nw in nnw:
            uu.append(mod.reshape(uw[s:s + np.prod(nw)], nw))
            s += np.prod(nw)

        return [
            MultigridDecomp.interp_field(u, i, mod=mod)
            for i, u in enumerate(uu)
        ]

    @staticmethod
    def get_weights_coarse_ones(nnw, dtype, mod):
        res = []
        s = 0
        for i, nw in enumerate(nnw):
            if i == len(nnw) - 1:
                res.append(mod.ones(np.prod(nw), dtype=dtype))
            else:
                res.append(mod.zeros(np.prod(nw), dtype=dtype))
            s += np.prod(nw)

        return mod.concat(res, -1)

    @staticmethod
    def weights_to_field(uw, nnw, mod):
        uu = []
        s = 0
        for nw in nnw:
            uu.append(mod.reshape(uw[s:s + np.prod(nw)], nw))
            s += np.prod(nw)

        res = None
        for i, u in reversed(list(enumerate(uu))):
            if res is None:
                res = u
                continue
            res = MultigridDecomp.interp_field(res, mod=mod) + u
        return res
