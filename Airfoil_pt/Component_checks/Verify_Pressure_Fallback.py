"""Discrete-operator consistency test; not an airfoil accuracy test."""
from pathlib import Path
import numpy as np
source = Path(__file__).with_name('Airfoil.py')
ns = {'__file__': str(source), '__name__': 'pressure_test_setup'}
exec(compile(source.read_text().split('# Main Cfd Loop')[0], str(source), 'exec'), ns)
core = ns['core']
fallback = ns['pressure_fallback']
rng = np.random.default_rng(42)
exact = rng.normal(size=ns['P'].shape)
ns['apply_pressure_bc'](exact)
rhs = core.poisson_residual(exact, np.zeros_like(core.fluid_center, dtype=float))
solved, residual = fallback.solve(np.zeros_like(exact), rhs)
error = np.max(np.abs((solved-exact)[1:-1,1:-1][core.fluid_center]))
print(f'Discrete pressure recovery max error: {error:.6e}')
print(f'Original stencil residual: {residual:.6e}')
assert error < 1e-7
print('PASS: sparse solve matches the original masked pressure operator and boundary conditions.')
