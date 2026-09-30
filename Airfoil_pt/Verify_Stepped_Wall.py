"""Smooth manufactured diffusion test on an L-shaped fluid domain.
Run beside cfd_core.py. Requires numpy, scipy, matplotlib.
Tests the current diffusion stencil, including mixed solid/fluid neighbors.
Not a coupled Navier-Stokes, corner-singularity, or airfoil force validation.
Exact streamfunction psi=f(x)f(y), f(s)=s^2(s-.5)^2(s-1)^2.
u=psi_y, v=-psi_x: analytic divergence zero and no slip on all walls.
Steady diffusion forcing is -Laplacian(u), -Laplacian(v).
The sparse matrix is checked against diffusion extracted from the shared
predictor by subtracting predictions with and without viscosity.
"""
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from cfd_core import CFDCore

f=Polynomial.fromroots([0,0,.5,.5,1,1])
d=[f.deriv(k) for k in range(4)]

def matrix_for(active, count, h):
    ids=np.full(active.shape,-1,int);ids[active]=np.arange(active.sum())
    rr=[];cc=[];vv=[]
    for i,j in zip(*np.nonzero(active)):
        k=ids[i,j];rr.append(k);cc.append(k);vv.append((4+count[i,j])/h**2)
        for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            a,b=i+di,j+dj
            if 0<=a<active.shape[0] and 0<=b<active.shape[1] and active[a,b]:
                rr.append(k);cc.append(ids[a,b]);vv.append(-1/h**2)
    return coo_matrix((vv,(rr,cc)),shape=(active.sum(),active.sum())).tocsr()

if __name__=='__main__':
    records=[]
    for n in [16,32,64,128]:
        h=1/n;coord=(np.arange(n+2)-.5)*h;X,Y=np.meshgrid(coord,coord)
        fluid=(X>0)&(X<1)&(Y>0)&(Y<1)&~((X>.5)&(Y>.5));solid=~fluid
        uf=fluid[:,:-1]&fluid[:,1:];vf=fluid[:-1,:]&fluid[1:,:]
        ua=uf[1:-1,1:];va=vf[1:,1:-1]
        uc=(ua&solid[:-2,1:-1]&solid[:-2,2:]).astype(float)+(ua&solid[2:,1:-1]&solid[2:,2:]).astype(float)
        vc=(va&solid[1:-1,:-2]&solid[2:,:-2]).astype(float)+(va&solid[1:-1,2:]&solid[2:,2:]).astype(float)
        um=ua&((solid[:-2,1:-1]^solid[:-2,2:])|(solid[2:,1:-1]^solid[2:,2:]))
        vm=va&((solid[1:-1,:-2]^solid[2:,:-2])|(solid[1:-1,2:]^solid[2:,2:]))
        # Pressure is not used in this diffusion-only component test.
        z=np.zeros((n,n));b=np.zeros((n,n),bool)
        core=CFDCore(dx=h,dy=h,dt=1.,dt_nu=1.,inv_dx2=h**-2,inv_dy2=h**-2,
            u_wall_count=uc,v_wall_count=vc,a_e=z,a_w=z,a_n=z,a_s=z,
            pressure_denominator=z,fluid_center=fluid[1:-1,1:-1],red_cells=b,
            black_cells=b,pressure_max_iterations=1,pressure_residual_tolerance=1e-8,
            sor_omega=1.,apply_pressure_bc=lambda p:None)
        matrices=[matrix_for(ua,uc,h),matrix_for(va,vc,h)]
        rng=np.random.default_rng(7);U=np.zeros_like(X);V=np.zeros_like(X)
        U[1:-1,1:-1][ua]=rng.normal(size=ua.sum())
        V[1:-1,1:-1][va]=rng.normal(size=va.sum())
        visc=core.predict_velocity(U,V);core.dt_nu=0.;invisc=core.predict_velocity(U,V)
        mismatch=0.
        for A,q,a,p1,p0 in zip(matrices,[U,V],[ua,va],visc,invisc):
            lhs=A@q[1:-1,1:-1][a]; extracted=-(p1-p0)[1:-1,1:-1][a]
            mismatch=max(mismatch,np.max(np.abs(lhs-extracted))/max(1.,np.max(abs(lhs))))
        assert mismatch<1e-12, 'Matrix differs from shared diffusion stencil'
        errors=[];exact_values=[];corner_errors=[]
        for component,A,a,m in zip(['U','V'],matrices,[ua,va],[um,vm]):
            xx=X[1:-1,1:-1]+(h/2 if component=='U' else 0)
            yy=Y[1:-1,1:-1]+(h/2 if component=='V' else 0)
            if component=='U':
                exact=d[0](xx)*d[1](yy);forcing=-(d[2](xx)*d[1](yy)+d[0](xx)*d[3](yy))
            else:
                exact=-d[1](xx)*d[0](yy);forcing=d[3](xx)*d[0](yy)+d[1](xx)*d[2](yy)
            numerical=spsolve(A,forcing[a]);err=numerical-exact[a]
            errors.extend(err);exact_values.extend(exact[a]);corner_errors.extend(err[m[a]])
        relative=100*np.linalg.norm(errors)/np.linalg.norm(exact_values)
        corner=max(abs(np.asarray(corner_errors)))/max(abs(np.asarray(exact_values)))*100
        records.append([n,relative,corner,mismatch,int(um.sum()+vm.sum())])
        print(f'{n:3d}: velocity relative RMS={relative:.6f}%; mixed-stencil max error / global peak={corner:.6f}%; operator match={mismatch:.2e}; mixed faces={int(um.sum()+vm.sum())}',flush=True)
    data=np.asarray(records)
    for i in range(1,len(data)):
        print(f'Observed velocity refinement order: {np.log2(data[i-1,1]/data[i,1]):.3f}')
    out=Path(__file__).resolve().parent/'stepped_wall_results';out.mkdir(exist_ok=True)
    np.savetxt(out/'errors.csv',data,delimiter=',',header='cells_per_side,velocity_relative_RMS_percent,mixed_max_error_over_global_peak_percent,operator_relative_mismatch,mixed_faces',comments='')
    fig,ax=plt.subplots(figsize=(8,5))
    ax.loglog(data[:,0],data[:,1],'o-',label='Velocity RMS error (%)')
    ax.loglog(data[:,0],data[:,2],'s--',label='Mixed-face max error / global peak (%)')
    ax.set(xlabel='Cells per unit length',ylabel='Error (%)',title='Smooth stepped-wall diffusion verification')
    ax.grid(True,which='both',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'convergence.png',dpi=180)
    print('Smooth diffusion component only; no pressure/advection coupling or singular corner solution tested.')
    plt.show()
