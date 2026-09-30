"""Sparse fallback for the airfoil's specific homogeneous pressure boundaries.
Left, bottom, top: zero normal gradient. Right: zero pressure.
Not suitable for the periodic vortex boundary conditions.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

class AirfoilPressureFallback:
    def __init__(self, core):
        self.core = core
        self.factor = None
        self.calls = 0

    def build(self):
        c = self.core
        mask = c.fluid_center
        ids = np.full(mask.shape, -1, dtype=int)
        ids[mask] = np.arange(mask.sum())
        rows, cols, vals = [], [], []
        h, w = mask.shape
        for i, j in zip(*np.nonzero(mask)):
            k = ids[i,j]
            diagonal = 0.0
            for di,dj,a in [(0,1,c.a_e),(0,-1,c.a_w),(1,0,c.a_n),(-1,0,c.a_s)]:
                coeff = a[i,j]
                if coeff == 0: continue
                ni,nj = i+di,j+dj
                if 0 <= ni < h and 0 <= nj < w:
                    if ids[ni,nj] < 0: raise ValueError('Coefficient points into solid')
                    rows.append(k);cols.append(ids[ni,nj]);vals.append(-coeff)
                    diagonal += coeff
                elif nj == w:
                    diagonal += coeff  # homogeneous outlet Dirichlet
                # Other external neighbors equal the center (Neumann).
            rows.append(k);cols.append(k);vals.append(diagonal)
        matrix = coo_matrix((vals,(rows,cols)),shape=(mask.sum(),mask.sum())).tocsc()
        self.factor = splu(matrix)

    def solve(self, pressure, rhs):
        if self.factor is None:
            print('Building pressure fallback factorization...', flush=True)
            self.build()
        c = self.core
        pressure[1:-1,1:-1][c.fluid_center] = self.factor.solve(-rhs[c.fluid_center])
        c.apply_pressure_bc(pressure)
        residual = float(np.max(np.abs(c.poisson_residual(pressure,rhs)[c.fluid_center])))
        if not np.isfinite(residual) or residual >= c.pressure_residual_tolerance:
            raise RuntimeError(f'Pressure fallback failed: residual={residual:.6e}; step rejected')
        self.calls += 1
        return pressure, residual
