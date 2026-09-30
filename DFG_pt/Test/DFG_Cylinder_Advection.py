"""DFG 2D-1 cylinder: standalone Cartesian benchmark implementation.

Run this file in VS Code. Dependencies: numpy, scipy, matplotlib.

Install dependencies in the selected VS Code interpreter:
    python -m pip install --upgrade numpy scipy matplotlib
Default: 440 x 82 cells, conservative centered advection experiment.


Checked 440 x 82, dt=0.002 run (T=17.8, steady criterion met):
Conventional middle CV: Cd=5.612441623 (+0.590%), Cl=0.012020560 (+13.199%).
Pressure difference=0.111287834 (-5.303%): worse than donor baseline.
Cd CV spread=0.112%; Cl absolute CV spread=3.67525e-05.
Stencil Cd=5.612629299, Cl=0.011968834.
Smooth periodic advection component RMS errors at 32,64,128 cells:
0.0079794476, 0.0020046890, 0.0005017885 (approximately second order).
These tests do not establish mesh independence or whole-solver accuracy.

Reference and boundary convention:
https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
Stress convention here: rho*nu*grad(velocity) - p*I.
At outlet: nu*du/dx - p/rho = 0, dv/dx = 0.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt

# Grid counts are pressure CELLS, not the old airfoil's node counts.
NX = 440
NY = 82
DT = 0.002
FINAL_TIME = 20.0
SHOW_PLOTS = True
CHECK_TIME_INTERVAL = 0.2  # simulated time between convergence checks
STOP_WHEN_STEADY = True
MINIMUM_TIME = 5.0
REQUIRED_CHECKS = 5
VELOCITY_CHANGE_TOL = 1e-5  # full-field change / reference mean speed
COEFFICIENT_CHANGE_TOL = 1e-5  # absolute change in Cd and Cl per check
DIVERGENCE_TOL = 1e-8
LENGTH, HEIGHT = 2.2, 0.41
CX, CY, RADIUS = 0.2, 0.2, 0.05
RHO, NU, U_MAX = 1.0, 0.001, 0.3
U_MEAN = 2.0 * U_MAX / 3.0
DIAMETER = 2.0 * RADIUS
REFERENCE = {'Cd': 5.57953523384, 'Cl': 0.010618948146,
             'pressure_difference': 0.11752016697}


ADVECTION_SCHEME = "central"  # central or donor

def advective_predictor(U,V,dx,dy,dt):
 uc=U[1:-1,1:-1];vc=V[1:-1,1:-1]
 def f(a,l,r):
  return a*(0.5*(l+r) if ADVECTION_SCHEME == "central" else np.where(a>=0,l,r))
 ue=f(.5*(uc+U[1:-1,2:]),uc,U[1:-1,2:]);uw=f(.5*(U[1:-1,:-2]+uc),U[1:-1,:-2],uc)
 un=f(.5*(vc+V[1:-1,2:]),uc,U[2:,1:-1]);us=f(.5*(V[:-2,1:-1]+V[:-2,2:]),U[:-2,1:-1],uc)
 ve=f(.5*(uc+U[2:,1:-1]),vc,V[1:-1,2:]);vw=f(.5*(U[1:-1,:-2]+U[2:,:-2]),V[1:-1,:-2],vc)
 vn=f(.5*(vc+V[2:,1:-1]),vc,V[2:,1:-1]);vs=f(.5*(V[:-2,1:-1]+vc),V[:-2,1:-1],vc)
 return uc-dt*((ue-uw)/dx+(un-us)/dy),vc-dt*((ve-vw)/dx+(vn-vs)/dy)


class CylinderSolver:
    def __init__(self, nx, ny, dt, cylinder=True):
        if nx < 40 or ny < 12 or dt <= 0:
            raise ValueError('Use nx >= 40, ny >= 12, dt > 0.')
        self.nx, self.ny, self.dt = nx, ny, dt
        self.dx, self.dy = LENGTH/nx, HEIGHT/ny
        self.x = (np.arange(nx)+.5)*self.dx
        self.y = (np.arange(ny)+.5)*self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.solid = (((self.X-CX)**2+(self.Y-CY)**2) <= RADIUS**2) if cylinder else np.zeros((ny,nx),bool)
        self.fluid = ~self.solid
        self.uopen = np.ones((ny, nx+1), bool)
        self.uopen[:,1:-1] = self.fluid[:,:-1] & self.fluid[:,1:]
        self.vopen = np.zeros((ny+1,nx), bool)
        self.vopen[1:-1] = self.fluid[:-1] & self.fluid[1:]
        self.U = np.zeros((ny+2,nx+2)); self.V = np.zeros_like(self.U)
        self.P = np.zeros((ny,nx))
        self.inlet = 4*U_MAX*self.y*(HEIGHT-self.y)/HEIGHT**2
        self.uid = np.full((ny,nx+1),-1,int)
        self.vid = np.full((ny+1,nx),-1,int)
        self.pid = np.full((ny,nx),-1,int)
        umask = self.uopen.copy(); umask[:,0] = False
        self.uid[umask] = np.arange(umask.sum())
        self.nu_dof = int(umask.sum())
        self.vid[self.vopen] = self.nu_dof+np.arange(self.vopen.sum())
        self.nv_dof = int(self.vopen.sum())
        self.pid[self.fluid] = self.nu_dof+self.nv_dof+np.arange(self.fluid.sum())
        self.size = self.nu_dof+self.nv_dof+int(self.fluid.sum())
        self.uwhere = np.nonzero(self.uid >= 0); self.vwhere = np.nonzero(self.vid >= 0)
        self.apply_ghosts()
        self.build_matrix()

    def apply_ghosts(self):
        U,V=self.U,self.V
        U[1:-1,:self.nx+1][~self.uopen] = 0
        V[:self.ny+1,1:-1][~self.vopen] = 0
        U[1:-1,0]=self.inlet
        # Tangential channel-wall ghost values impose zero at the wall face.
        U[0,:]=-U[1,:]; U[-1,:]=-U[-2,:]
        V[:,0]=-V[:,1]  # zero transverse velocity at inlet, halfway to ghost
        V[:,-1]=V[:,-2]  # outlet dv/dx=0
        V[-1,:]=-V[-3,:]
        U[:,-1]=2*U[:,-2]-U[:,-3]  # unused outlet predictor extension

    def build_matrix(self):
        nx,ny,dx,dy,dt=self.nx,self.ny,self.dx,self.dy,self.dt
        rows=[];cols=[];vals=[];self.fixed_rhs=np.zeros(self.size)
        def add(r,c,v):
            if c < 0: raise ValueError('Unexpected inactive unknown')
            rows.append(r);cols.append(c);vals.append(v)
        for i,j in zip(*self.uwhere):
            row=self.uid[i,j]
            if j==nx:
                # Outlet normal traction: 2nd-order backward du/dx and
                # linear extrapolation of pressure to x=L. No p=0 constraint.
                add(row,row,3*NU/(2*dx));add(row,self.uid[i,j-1],-2*NU/dx)
                add(row,self.uid[i,j-2],NU/(2*dx))
                add(row,self.pid[i,-1],-1.5/RHO);add(row,self.pid[i,-2],.5/RHO)
                continue
            diagonal=1.0
            for di,dj,h in [(0,-1,dx),(0,1,dx),(-1,0,dy),(1,0,dy)]:
                ni,nj=i+di,j+dj; a=dt*NU/h**2
                if ni<0 or ni>=ny:
                    diagonal+=2*a  # reflected tangential wall ghost
                elif nj==0:
                    diagonal+=a; self.fixed_rhs[row]+=a*self.inlet[ni]
                elif self.uid[ni,nj]>=0:
                    diagonal+=a;add(row,self.uid[ni,nj],-a)
                else:
                    diagonal+=a
                    # Same AND rule for fully solid transverse neighbors.
                    if di and self.solid[ni,j-1] and self.solid[ni,j]: diagonal+=a
            add(row,row,diagonal)
            add(row,self.pid[i,j],dt/(RHO*dx));add(row,self.pid[i,j-1],-dt/(RHO*dx))
        for i,j in zip(*self.vwhere):
            row=self.vid[i,j]; diagonal=1.0
            for di,dj,h in [(0,-1,dx),(0,1,dx),(-1,0,dy),(1,0,dy)]:
                ni,nj=i+di,j+dj; a=dt*NU/h**2
                if nj<0: diagonal+=2*a  # zero v at inlet
                elif nj>=nx: pass  # homogeneous Neumann outlet ghost equals center
                elif ni==0 or ni==ny: diagonal+=a  # known wall normal velocity
                elif self.vid[ni,nj]>=0:
                    diagonal+=a;add(row,self.vid[ni,nj],-a)
                else:
                    diagonal+=a
                    if dj and self.solid[i-1,nj] and self.solid[i,nj]: diagonal+=a
            add(row,row,diagonal)
            add(row,self.pid[i,j],dt/(RHO*dy));add(row,self.pid[i-1,j],-dt/(RHO*dy))
        for i,j in zip(*np.nonzero(self.fluid)):
            row=self.pid[i,j]
            for ids,ii,jj,coeff in [(self.uid,i,j+1,1/dx),(self.uid,i,j,-1/dx),
                                    (self.vid,i+1,j,1/dy),(self.vid,i,j,-1/dy)]:
                k=ids[ii,jj]
                if k>=0: add(row,k,coeff)
                elif ids is self.uid and jj==0: self.fixed_rhs[row]-=coeff*self.inlet[ii]
        self.matrix=coo_matrix((vals,(rows,cols)),shape=(self.size,self.size)).tocsc()
        print(f'Factoring coupled system: {self.size:,} unknowns...',flush=True)
        self.factor=splu(self.matrix)
        print('Factorization complete. Advancing flow...',flush=True)

    def step(self):
        self.apply_ghosts()
        up,vp=advective_predictor(self.U,self.V,self.dx,self.dy,self.dt)
        self.last_U=self.U.copy(); self.last_V=self.V.copy()
        self.last_adU=(self.U[1:-1,1:-1]-up)/self.dt
        self.last_adV=(self.V[1:-1,1:-1]-vp)/self.dt
        rhs=self.fixed_rhs.copy()
        i,j=self.uwhere; internal=j<self.nx
        rhs[self.uid[i[internal],j[internal]]] += up[i[internal],j[internal]-1]
        i,j=self.vwhere;rhs[self.vid[i,j]] += vp[i-1,j]
        solution=self.factor.solve(rhs)
        residual=float(np.max(np.abs(self.matrix@solution-rhs)))
        if not np.all(np.isfinite(solution)) or residual>1e-7:
            raise RuntimeError(f'Coupled linear solve failed: residual {residual:.3e}')
        i,j=self.uwhere;self.U[i+1,j]=solution[self.uid[i,j]]
        i,j=self.vwhere;self.V[i,j+1]=solution[self.vid[i,j]]
        self.P[self.fluid]=solution[self.pid[self.fluid]]
        self.apply_ghosts()
        self.linear_residual=residual

    def centered(self):
        return .5*(self.U[1:-1,:-2]+self.U[1:-1,1:-1]), .5*(self.V[:-2,1:-1]+self.V[1:-1,1:-1])

    def diagnostics(self):
        div=(self.U[1:-1,1:-1]-self.U[1:-1,:-2])/self.dx+(self.V[1:-1,1:-1]-self.V[:-2,1:-1])/self.dy
        qi=np.sum(self.U[1:-1,0])*self.dy;qo=np.sum(self.U[1:-1,-2])*self.dy
        flux=qo-qi; integral=np.sum(div[self.fluid])*self.dx*self.dy
        normal=NU*(3*self.U[1:-1,-2]-4*self.U[1:-1,-3]+self.U[1:-1,-4])/(2*self.dx)-(1.5*self.P[:,-1]-.5*self.P[:,-2])/RHO
        tangential=NU*(self.V[:self.ny+1,-1]-self.V[:self.ny+1,-2])/self.dx
        u,v=self.centered()
        cfl=self.dt*(np.max(abs(self.U))/self.dx+np.max(abs(self.V))/self.dy)
        return dict(max_divergence=float(np.max(abs(div[self.fluid]))),Qin=float(qi),Qout=float(qo),
                    relative_mass_error=float(flux/qi),flux_divergence_mismatch=float(flux-integral),
                    outlet_normal_residual=float(np.max(abs(normal))),outlet_tangent_residual=float(np.max(abs(tangential))),
                    cfl=float(cfl),max_speed=float(np.max(np.hypot(u,v))),linear_residual=self.linear_residual)

    def force(self, previous, bounds):
        u,v=self.centered();old_u,old_v=previous
        # Values inside the represented solid are zero in the storage integral.
        u=u.copy();v=v.copy();u[self.solid]=0;v[self.solid]=0
        du=(u-old_u)/self.dt;dv=(v-old_v)/self.dt
        ux=np.gradient(u,self.dx,axis=1);uy=np.gradient(u,self.dy,axis=0)
        vx=np.gradient(v,self.dx,axis=1);vy=np.gradient(v,self.dy,axis=0)
        l,r,b,t=[int(np.argmin(abs(a-target))) for a,target in zip([self.x,self.x,self.y,self.y],bounds)]
        if self.solid[b:t+1,l].any() or self.solid[b:t+1,r].any() or self.solid[b,l:r+1].any() or self.solid[t,l:r+1].any():
            raise ValueError('Control-volume boundary intersects cylinder mask')
        xx=self.x[l:r+1];yy=self.y[b:t+1];p=self.P
        trap=np.trapezoid
        fx=trap(p[b:t+1,l],yy)-trap(p[b:t+1,r],yy)
        fy=trap(p[b,l:r+1],xx)-trap(p[t,l:r+1],xx)
        # Unsymmetrized viscous stress matches the benchmark's convention.
        fx+=RHO*NU*(trap(ux[b:t+1,r]-ux[b:t+1,l],yy)+trap(uy[t,l:r+1]-uy[b,l:r+1],xx))
        fy+=RHO*NU*(trap(vx[b:t+1,r]-vx[b:t+1,l],yy)+trap(vy[t,l:r+1]-vy[b,l:r+1],xx))
        fx+=RHO*(trap(u[b:t+1,l]**2-u[b:t+1,r]**2,yy)+trap((u*v)[b,l:r+1]-(u*v)[t,l:r+1],xx))
        fy+=RHO*(trap((u*v)[b:t+1,l]-(u*v)[b:t+1,r],yy)+trap(v[b,l:r+1]**2-v[t,l:r+1]**2,xx))
        unsteady_x=-RHO*trap(trap(du[b:t+1,l:r+1],xx,axis=1),yy)
        unsteady_y=-RHO*trap(trap(dv[b:t+1,l:r+1],xx,axis=1),yy)
        fx+=unsteady_x;fy+=unsteady_y
        scale=.5*RHO*U_MEAN**2*DIAMETER
        return dict(Cd=float(fx/scale),Cl=float(fy/scale),drag=float(fx),lift=float(fy),
                    Fx_unsteady=float(unsteady_x),Fy_unsteady=float(unsteady_y),
                    bounds=[float(self.x[l]),float(self.x[r]),float(self.y[b]),float(self.y[t])])

    def stencil_force(self, bounds):

        l,r,b,t=[int(np.argmin(abs(a-target))) for a,target in zip([self.x,self.x,self.y,self.y],bounds)]
        sy,sx=np.nonzero(self.solid)
        if not (l>1 and r<self.nx-2 and b>1 and t<self.ny-2 and
                l+2<sx.min() and r-2>sx.max() and b+2<sy.min() and t-2>sy.max()):
            raise ValueError('Stencil force requires a rectangle enclosing the solid with fluid margin.')
        p=np.pad(self.P,1)
        gradients=[(p[1:-1,2:]-p[1:-1,1:-1])/self.dx,
                   (p[2:,1:-1]-p[1:-1,1:-1])/self.dy]
        forces=[]
        for old,new,ad,gradient in zip([self.last_U,self.last_V],[self.U,self.V],
                                      [self.last_adU,self.last_adV],gradients):
            c=new[1:-1,1:-1]
            lap=(new[1:-1,2:]-2*c+new[1:-1,:-2])/self.dx**2
            lap+=(new[2:,1:-1]-2*c+new[:-2,1:-1])/self.dy**2
            residual=(c-old[1:-1,1:-1])/self.dt+ad-NU*lap+gradient/RHO
            forces.append(float(-RHO*np.sum(residual[b:t+1,l:r+1])*self.dx*self.dy))
        scale=.5*RHO*U_MEAN**2*DIAMETER
        return {'Cd':forces[0]/scale,'Cl':forces[1]/scale,
                'drag':forces[0],'lift':forces[1]}

    def pressure_probe(self, front, radius_cells=3.0):
        # Local planar extrapolation to the actual circular boundary point.
        # Explicitly reported as an estimate; solid pressure is never sampled.
        px=CX-RADIUS if front else CX+RADIUS;py=CY
        h=max(self.dx,self.dy)
        exterior=self.X<=px if front else self.X>=px
        distance=np.hypot(self.X-px,self.Y-py)
        mask=self.fluid & exterior & (distance<radius_cells*h)
        A=np.column_stack([np.ones(mask.sum()),(self.X[mask]-px)/h,(self.Y[mask]-py)/h])
        if len(A)<3 or np.linalg.matrix_rank(A)<3: raise ValueError('Pressure probe lacks fluid support')
        fit=np.linalg.lstsq(A,self.P[mask],rcond=None)[0]
        return float(fit[0])


def plots(s,history,out):
    u,v=s.centered();speed=np.ma.masked_where(s.solid,np.hypot(u,v));p=np.ma.masked_where(s.solid,s.P)
    plt.rcParams.update({'font.size':11})
    fig,axes=plt.subplots(2,1,figsize=(12,6),layout='constrained')
    for ax,z,title in zip(axes,[speed,p],['Speed','Pressure']):
        artist=ax.pcolormesh(s.x,s.y,z,shading='nearest',cmap='viridis' if title=='Speed' else 'coolwarm')
        fig.colorbar(artist,ax=ax,label=title)
        ax.add_patch(plt.Circle((CX,CY),RADIUS,color='0.2'))
        ax.set(xlim=(0,LENGTH),ylim=(0,HEIGHT),xlabel='x',ylabel='y',title=title);ax.set_aspect('equal')
    fig.suptitle('DFG cylinder configuration · Re 20 · Cartesian approximation')
    fig.savefig(out/'flow.png',dpi=170)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    times=[r['T'] for r in history]
    for ax,key in zip(axes.flat,['Cd','Cl','max_divergence','relative_mass_error']):
        values=np.array([r[key] for r in history])
        if key in REFERENCE:
            ax.plot(times,values,label='Conventional CV estimate');ax.axhline(REFERENCE[key],color='darkorange',ls='--',label='Published reference');ax.legend()
        else:
            ax.semilogy(times,np.maximum(abs(values),1e-16))
            ax.set_ylim(1e-16, max(1e-7, 10*np.max(abs(values))))
            ax.set_ylabel('Absolute '+key.replace('_',' '))
        ax.set_xlabel('Simulated time')
        if key in REFERENCE: ax.set_ylabel(key)
        ax.grid(alpha=.25)
    fig.suptitle('Force development and numerical diagnostics · agreement is not assumed')
    fig.savefig(out/'history.png',dpi=170)


def pressure_audit(x, y, pressure, solid, out):
    X, Y = np.meshgrid(x, y)
    h = max(float(x[1]-x[0]), float(y[1]-y[0]))
    rows = []
    for radius in (3., 4., 5.):
        for degree in (1, 2):
            fits = []
            for px, sign in ((CX-RADIUS, -1), (CX+RADIUS, 1)):
                a, b = (X-px)/h, (Y-CY)/h
                mask = (~solid.astype(bool)) & (sign*(X-px)>=0) & (a*a+b*b<radius**2)
                a, b = a[mask], b[mask]
                columns = [np.ones_like(a), a, b]
                if degree == 2:
                    columns += [a*a, a*b, b*b]
                A = np.column_stack(columns)
                fit, _, rank, _ = np.linalg.lstsq(A, pressure[mask], rcond=None)
                if rank != A.shape[1]:
                    raise ValueError('Pressure audit fit is rank deficient')
                fits.append((float(fit[0]), float(np.linalg.cond(A)), len(a)))
            rows.append(dict(radius_cells=radius, degree=degree,
                pressure_difference=fits[0][0]-fits[1][0],
                front_samples=fits[0][2], rear_samples=fits[1][2],
                max_condition=max(fits[0][1], fits[1][1])))
    with (out/'pressure_audit.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
    for degree, label in ((1,'Linear extrapolation'), (2,'Quadratic extrapolation')):
        selected=[r for r in rows if r['degree']==degree]
        ax.plot([r['radius_cells'] for r in selected],
                [r['pressure_difference'] for r in selected], 'o-', label=label)
    ax.axhline(REFERENCE['pressure_difference'], color='black', ls='--', label='Published reference')
    ax.set(xlabel='Sampling radius / grid spacing', ylabel='Front minus rear pressure',
           title='Pressure sampling sensitivity — fixed choices, no best-fit selection')
    ax.legend(); ax.grid(alpha=.25)
    fig.savefig(out/'pressure_audit.png', dpi=170)
    print('Pressure audit (does not change the solved field):')
    for row in rows:
        print(f"radius={row['radius_cells']:g}h, degree={row['degree']}: delta P={row['pressure_difference']:.9f}")
    return rows


def run():
    global ADVECTION_SCHEME
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx',type=int,default=NX);parser.add_argument('--ny',type=int,default=NY)
    parser.add_argument('--dt',type=float,default=DT);parser.add_argument('--time',type=float,default=FINAL_TIME)
    parser.add_argument('--no-show',action='store_true');parser.add_argument('--no-early-stop',action='store_true')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--advection', choices=['central','donor'], default=ADVECTION_SCHEME)
    parser.add_argument('--analyze', type=Path, help='Audit a saved flow NPZ without running the solver')
    args=parser.parse_args()
    if args.analyze is not None:
        out = args.output or args.analyze.resolve().parent/(args.analyze.stem+'_pressure_audit')
        out.mkdir(parents=True, exist_ok=True)
        with np.load(args.analyze, allow_pickle=False) as saved:
            if 'method' not in saved or 'coupled_implicit_diffusion_conservative_' not in str(saved['method']):
                parser.error('Expected an NPZ from this DFG conservative solver')
            print('Saved method:', str(saved['method']), flush=True)
            pressure_audit(saved['x'], saved['y'], saved['P'], saved['solid'], out)
        print('Saved audit:', out.resolve())
        if SHOW_PLOTS and not args.no_show: plt.show()
        return
    ADVECTION_SCHEME = args.advection
    if not np.isfinite(args.time) or not np.isfinite(args.dt) or args.time<=0 or args.dt<=0:
        parser.error('time and dt must be finite and positive')
    steps=int(np.ceil(args.time/args.dt));dt=args.time/steps
    check_every = max(1, int(round(CHECK_TIME_INTERVAL / dt)))
    out=args.output or Path(__file__).resolve().parent/f'dfg_{ADVECTION_SCHEME}_{args.nx}x{args.ny}_dt{dt:g}'
    out.mkdir(parents=True,exist_ok=True)
    print(f'DFG 2D-1 | Re={U_MEAN*DIAMETER/NU:g} | cells={args.nx}x{args.ny} | dt={dt:g}',flush=True)
    print(f'Conservative {ADVECTION_SCHEME} advection; original airfoil solver is unchanged.',flush=True)
    print(f'Convergence checks every {check_every} steps (T interval={check_every*dt:g}); early stop={STOP_WHEN_STEADY and not args.no_early_stop}',flush=True)
    start=time.perf_counter();s=CylinderSolver(args.nx,args.ny,dt)
    cvs=[(.11,.31,.11,.29),(.09,.33,.09,.31),(.07,.35,.07,.33)]
    history=[];count=0;stopped=False;last_fields=None
    for step in range(1,steps+1):
        previous=tuple(a.copy() for a in s.centered())
        for a in previous: a[s.solid]=0
        s.step()
        if step<=3 or step % check_every==0 or step==steps:
            diag=s.diagnostics()
            if diag['cfl']>.8 or diag['max_divergence']>DIVERGENCE_TOL:
                raise RuntimeError(f'Step rejected by diagnostic gate: {diag}. Reduce dt if CFL exceeds 0.8.')
            if step % check_every==0 or step==steps:
                force=s.force(previous,cvs[1]);current=s.centered()
                change=float('inf') if last_fields is None else max(np.max(abs(a-b)) for a,b in zip(current,last_fields))/U_MEAN
                force_change=float('inf') if not history else max(abs(force[k]-history[-1][k]) for k in ['Cd','Cl'])
                count=count+1 if step*dt>=MINIMUM_TIME and change<VELOCITY_CHANGE_TOL and force_change<COEFFICIENT_CHANGE_TOL else 0
                record=dict(T=step*dt,**diag,**{k:force[k] for k in ['Cd','Cl','Fx_unsteady','Fy_unsteady']},velocity_change=change,coefficient_change=force_change)
                history.append(record);last_fields=tuple(a.copy() for a in current)
                print(f'Step {step:6d}/{steps} | T={step*dt:.3f} | Cd={force["Cd"]:.6f} | Cl={force["Cl"]:.6f} | div={diag["max_divergence"]:.2e} | mass={diag["relative_mass_error"]:.2e} | field change={change:.2e} | steady={count}/{REQUIRED_CHECKS}',flush=True)
                if STOP_WHEN_STEADY and not args.no_early_stop and count>=REQUIRED_CHECKS:
                    stopped=True;break
            else: print(f'Startup {step}: coupled residual={diag["linear_residual"]:.2e}',flush=True)
    results=[s.force(previous,cv) for cv in cvs]
    stencil_results=[s.stencil_force(cv) for cv in cvs]
    pd=s.pressure_probe(True)-s.pressure_probe(False)
    pd_wide=s.pressure_probe(True,4)-s.pressure_probe(False,4)
    summary=dict(advection_scheme=ADVECTION_SCHEME,grid_cells=[args.nx,args.ny],dx=s.dx,dy=s.dy,dt=dt,T=step*dt,steps=step,
                 Re=U_MEAN*DIAMETER/NU,steady_stop=stopped,steady_checks=count,diagnostics=s.diagnostics(),
                 Cd_relative_CV_spread=float(np.ptp([r['Cd'] for r in results])/abs(np.mean([r['Cd'] for r in results]))),
                 Cl_absolute_CV_spread=float(np.ptp([r['Cl'] for r in results])),
                 control_volumes=results,stencil_control_volumes=stencil_results,
                 stencil_Cd_spread=float(np.ptp([r['Cd'] for r in stencil_results])),
                 stencil_Cl_spread=float(np.ptp([r['Cl'] for r in stencil_results])),
                 pressure_difference_estimate=pd,pressure_difference_wider_stencil=pd_wide,
                 reference=REFERENCE,elapsed_seconds=time.perf_counter()-start,
                 limitations='Stair-step circle, explicit first-order time integration, extrapolated wall pressure. Centered advection is experimental; no whole-solver order claim. New coupled solve; not validation of the original projection solver.')
    print('\nFinal comparison (not an automatic benchmark pass):')
    for k,value in [('Cd',results[1]['Cd']),('Cl',results[1]['Cl']),('pressure_difference',pd)]:
        ref=REFERENCE[k];print(f'{k}: computed={value:.9f}, reference={ref:.9f}, relative difference={100*(value-ref)/abs(ref):+.3f}%')
    print(f'Cd control-volume spread: {100*summary["Cd_relative_CV_spread"]:.3f}%')
    print(f'Cl absolute control-volume spread: {summary["Cl_absolute_CV_spread"]:.6g}')
    print('Stencil-balance force estimates (separate from conventional CV):')
    for label,r in zip(['Inner','Middle','Outer'],stencil_results):
        print(f'{label}: Cd={r["Cd"]:.9f}, Cl={r["Cl"]:.9f}')
    print(f'Stencil Cd spread: {summary["stencil_Cd_spread"]:.3e}; not an accuracy bound')
    print(f'Pressure probe stencil sensitivity: {abs(pd-pd_wide):.3e}')
    print(f'Steady criterion met: {count>=REQUIRED_CHECKS}; elapsed {summary["elapsed_seconds"]:.1f}s')
    with (out/'summary.json').open('w') as f: json.dump(summary,f,indent=2)
    with (out/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    u,v=s.centered()
    np.savez_compressed(out/'flow.npz',x=s.x,y=s.y,U_face=s.U,V_face=s.V,U_cell=u,V_cell=v,P=s.P,solid=s.solid,dt=dt,T=step*dt,Re=U_MEAN*DIAMETER/NU,pressure_is_physical=True,
                        rho=RHO,nu=NU,U_mean=U_MEAN,diameter=DIAMETER,
                        method='coupled_implicit_diffusion_conservative_'+ADVECTION_SCHEME+'_flux',
                        x_u=np.arange(s.nx+1)*s.dx,y_v=np.arange(s.ny+1)*s.dy)
    pressure_audit(s.x,s.y,s.P,s.solid,out)
    plots(s,history,out)
    print('Saved results:',out.resolve())
    if SHOW_PLOTS and not args.no_show: plt.show()


if __name__=='__main__':
    run()
