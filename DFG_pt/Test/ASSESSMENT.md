# DFG Re20 steady sensitivity assessment

Completed using the unchanged central-advection cylinder solver. The original airfoil solver is separate.

## Results

| Case | Cd | Cl | Pressure difference | Steady checks |

| 440 × 82, dt=.002 | 5.612441623 | 0.012020560 | 0.111287834 | 5/5 |

| 440 × 82, dt=.001 | 5.612441623 | 0.012020563 | 0.111287834 | 5/5 |

| 660 × 123, dt=.002 | 5.635699205 | 0.011763987 | 0.115172529 | 5/5 |

| Published reference | 5.579535234 | 0.010618948 | 0.117520167 | — |

## Changes relative to the saved 440 × 82 baseline

| Comparison | Cd change | Cl change | Pressure difference change |

| 440 × 82, dt=.001 | -0.000000% | +0.000023% | -0.000000% |

| 660 × 123, dt=.002 | +0.414393% | -2.134458% | +3.490673% |

## Summary

 Grid sensitivity remains, and lift is still roughly 11–13% above the reference. The finer grid improves lift and the original pressure estimate but worsens drag agreement. 

## If you want to run it

Install numpy, scipy and matplotlib in your Python environment. The solver is included unchanged; no cfd_core.py is required. From this extracted directory:
```powershell
python Continue_Sensitivity.py --restart baseline_440_dt002/flow.npz --nx 440 --ny 82 --dt .001 --output repeat_timestep
python Continue_Sensitivity.py --restart baseline_440_dt002/flow.npz --nx 660 --ny 123 --dt .002 --output repeat_grid
```

Running DFG_Cylinder_Advection.py alone starts the full baseline simulation and shows charts. Repeating the packaged completed runs is optional.

## Source

[Official DFG 2D-1 benchmark](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html). Reference quantities and measurement points checked September 29, 2026.
