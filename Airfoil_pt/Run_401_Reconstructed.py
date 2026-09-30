# Reconstructed from uploaded settings and logs; original 401 runner was not supplied.
"""Run the current Airfoil.py on a finer grid without editing its file.
Keep this script beside Airfoil.py, cfd_core.py and pressure_fallback.py.
Default: 401x401, angle 0, dt=.00025, 20000 steps, T=5; charts enabled.
This runs the actual solver, not an independent copy of its algorithms.
Outputs go into a dedicated folder for this grid. No input NPZ is needed.
"""
import ast
from pathlib import Path
import sys
import time

GRID = 401
SHOW_CHARTS = True

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    source = root / 'Airfoil.py'
    if not source.exists():
        raise FileNotFoundError('Place Run_Grid_Check.py beside the current Airfoil.py.')
    sys.path.insert(0, str(root))
    tree = ast.parse(source.read_text(encoding='utf-8'))
    overrides = {'nx': GRID, 'ny': GRID, 'dt': 2.5e-4, 'num_steps': 20000,
                 'angle_of_attack': 0.0, 'SHOW_PLOTS': SHOW_CHARTS,
                 'STOP_AT_STEADY_STATE': False, 'steady_check_interval': 400}
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in overrides:
                node.value = ast.Constant(overrides[target.id]);found.add(target.id)
    if found != set(overrides):
        raise RuntimeError(f'Airfoil settings not found: {set(overrides)-found}')
    tree = ast.fix_missing_locations(tree)
    output = root / f'grid_check_{GRID}_dt_0p00025'
    output.mkdir(exist_ok=True)
    # Airfoil output paths are based on __file__; isolate this run's output.
    namespace = {'__name__': '__main__', '__file__': str(output / 'Airfoil.py')}
    print(f'Grid check: {GRID} x {GRID}, dt=0.00025, T=5, angle=0 degrees', flush=True)
    start = time.perf_counter()
    exec(compile(tree, str(source), 'exec'), namespace)
    elapsed = time.perf_counter()-start
    print(f'Elapsed execution time (including any time charts stayed open): {elapsed:.1f} seconds')
    baseline_cd = 0.79408263  # Previously verified 201x201 zero-angle run at T=5
    print(f"Cd change from recorded 201x201 baseline: {100*(namespace['Cd']-baseline_cd)/baseline_cd:+.6f}%")
    print('Compare force settling and CV spread as well as final Cd.')
    print('Two grids measure sensitivity; they do not establish grid independence.')
