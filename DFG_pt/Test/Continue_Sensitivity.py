import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--restart',type=Path,required=True)
parser.add_argument('--nx',type=int,required=True)
parser.add_argument('--ny',type=int,required=True)
parser.add_argument('--dt',type=float,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
solver_path=Path(__file__).with_name('DFG_Cylinder_Advection.py')
spec=importlib.util.spec_from_file_location('dfg',solver_path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original=m.CylinderSolver
source=np.load(args.restart,allow_pickle=False)
if str(source['method'])!='coupled_implicit_diffusion_conservative_central_flux':
 raise ValueError('Use the saved central-advection baseline, not donor or van Leer')
for key,value in [('rho',m.RHO),('nu',m.NU),('U_mean',m.U_MEAN),('diameter',m.DIAMETER)]:
 if not np.isclose(float(source[key]),value):raise ValueError('Restart physics mismatch: '+key)
X,Y=np.meshgrid(source['x'],source['y'])
expected_solid=(X-m.CX)**2+(Y-m.CY)**2<=m.RADIUS**2
if not np.array_equal(expected_solid,source['solid']):raise ValueError('Restart circle geometry mismatch')
if not np.isclose(source['x_u'][-1],m.LENGTH) or not np.isclose(source['y_v'][-1],m.HEIGHT):raise ValueError('Restart domain mismatch')
same=(len(source['x'])==args.nx and len(source['y'])==args.ny)
class Restarted(original):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw)
  if same:
   if not np.array_equal(self.solid,source['solid']):raise ValueError('Restart geometry mismatch')
   self.U[:]=source['U_face'];self.V[:]=source['V_face'];self.P[:]=source['P']
  else:
   def interp(y,x,field,yy,xx):
    Y,X=np.meshgrid(yy,xx,indexing='ij')
    return RegularGridInterpolator((y,x),field,bounds_error=False,fill_value=None)(np.stack([Y,X],axis=-1))
   self.U[1:-1,:self.nx+1]=interp(source['y'],source['x_u'],source['U_face'][1:-1,:len(source['x'])+1],self.y,np.arange(self.nx+1)*self.dx)
   self.V[:self.ny+1,1:-1]=interp(source['y_v'],source['x'],source['V_face'][:len(source['y'])+1,1:-1],np.arange(self.ny+1)*self.dy,self.x)
   self.P[:]=interp(source['y'],source['x'],source['P'],self.y,self.x)
  self.apply_ghosts()
m.CylinderSolver=Restarted
m.MINIMUM_TIME=1.0 if same else 3.0
sys.argv=[str(solver_path),'--nx',str(args.nx),'--ny',str(args.ny),'--dt',str(args.dt),'--time','12','--no-show','--output',str(args.output)]
print('Restarted STEADY assessment. T below is additional relaxation time.',flush=True)
m.run()
import json
p=args.output/'summary.json';data=json.loads(p.read_text())
data['initialization']={'type':'saved_steady_field' if same else 'interpolated_saved_field','source_sha256':hashlib.sha256(args.restart.read_bytes()).hexdigest(),'source_grid':[len(source['x']),len(source['y'])],'source_T':float(source['T']),'reported_T_is_additional_relaxation':True,'minimum_relaxation_time':m.MINIMUM_TIME}
data['solver_sha256']=hashlib.sha256(solver_path.read_bytes()).hexdigest()
p.write_text(json.dumps(data,indent=2))
