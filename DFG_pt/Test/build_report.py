"""Assemble the completed steady sensitivity assessment from saved summaries."""
from pathlib import Path
import json, shutil, hashlib, zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
base=root/'baseline_440_dt002';base.mkdir(exist_ok=True)
upload=root.parent/'upload'
for source,target in [('summary(2).json','summary.json'),('flow(2).npz','flow.npz'),('history(2).csv','history.csv')]:
 if not (base/target).exists(): shutil.copyfile(upload/source,base/target)
paths=[base,root/'restart_440_dt001',root/'restart_660_dt002']
data=[json.loads((p/'summary.json').read_text()) for p in paths]
labels=['440 × 82, dt=.002','440 × 82, dt=.001','660 × 123, dt=.002']
ref=data[0]['reference']
metrics=['Cd','Cl','pressure_difference']
values=[[d['control_volumes'][1]['Cd'],d['control_volumes'][1]['Cl'],d['pressure_difference_estimate']] for d in data]
lines=['# DFG Re20 steady sensitivity assessment','',
'Completed using the unchanged conservative central-advection cylinder solver. The original airfoil solver is separate.','',
'## Results','',
'| Case | Cd | Cl | Pressure difference | Steady checks |',
'|---|---:|---:|---:|---:|']
for label,d,vs in zip(labels,data,values):
 lines.append(f'| {label} | {vs[0]:.9f} | {vs[1]:.9f} | {vs[2]:.9f} | {d["steady_checks"]}/5 |')
lines += [f'| Published reference | {ref["Cd"]:.9f} | {ref["Cl"]:.9f} | {ref["pressure_difference"]:.9f} | — |','',
'## Changes relative to the saved 440 × 82 baseline','',
'| Comparison | Cd change | Cl change | Pressure difference change |','|---|---:|---:|---:|']
for i in (1,2):
 changes=100*(np.array(values[i])/np.array(values[0])-1)
 lines.append(f'| {labels[i]} | {changes[0]:+.6f}% | {changes[1]:+.6f}% | {changes[2]:+.6f}% |')
lines+=['','## Decision','',
'[Official DFG 2D-1 benchmark](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html)','']
(root/'ASSESSMENT.md').write_text('\n'.join(lines))
fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for k,ax in enumerate(axs):
 ax.plot(range(3),[v[k] for v in values],'o',label='Middle control volume' if k<2 else 'Original pressure estimate')
 ax.axhline(ref[metrics[k]],ls='--',color='black',label='Published reference')
 ax.set_xticks(range(3),['440\ndt=.002','440\ndt=.001','660\ndt=.002']);ax.set_title(metrics[k].replace('_',' '));ax.grid(alpha=.2)
 ax.legend(fontsize=8)
fig.suptitle('DFG Re20: steady timestep and grid sensitivity')
fig.savefig(root/'comparison.png',dpi=160)
files=[root/'DFG_Cylinder_Advection.py',root/'Continue_Sensitivity.py',root/'build_report.py',root/'ASSESSMENT.md',root/'comparison.png']
for p in paths:
 files+=list(p.glob('*'))
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
(root/'SHA256.json').write_text(json.dumps(manifest,indent=2));files.append(root/'SHA256.json')
with zipfile.ZipFile(root.parent/'DFG_Assessment.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in files:
  if p.is_file():z.write(p,'DFG_Assessment/'+str(p.relative_to(root)))
print('\n'.join(lines[:24]))
