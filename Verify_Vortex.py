"""Periodic 2D Taylor-Green test of the current CFD predictor/projection.
Imports the same cfd_core routines used by Airfoil.py.
Adapters: periodic ghost cells; no solid; zero-mean pressure gauge.
Airfoil boundary conditions, wall forces, and geometry are NOT tested.
Horizontal advection now selects the upstream neighbor by velocity sign.
Exact solution on [0,2*pi)^2:
u=sin(x)*cos(y)*exp(-2*nu*t), v=-cos(x)*sin(y)*exp(-2*nu*t),
p=rho/4*(cos(2*x)+cos(2*y))*exp(-4*nu*t).
The reference is sampled at each component's own staggered location.
"""
from pathlib import Path
from cfd_core import CFDCore
import numpy as np
import matplotlib.pyplot as plt


def periodic(a):
    a[0,1:-1]=a[-2,1:-1]; a[-1,1:-1]=a[1,1:-1]
    a[:,0]=a[:,-2]; a[:,-1]=a[:,1]


def apply_pressure_bc(P_field):
    periodic(P_field)






def backward_divergence(U_field, V_field):
    return core.backward_divergence(U_field, V_field)

def solve_pressure(P_field, rhs):
    return core.solve_pressure(P_field, rhs)

if __name__ == '__main__':
    rho=1.0
    nu=0.1
    end_time=0.1
    records=[]
    for n in (16,32,64):
        dx=dy=2*np.pi/n
        steps=n*2
        dt=end_time/steps
        dt_nu=nu*dt
        inv_dx2=1/dx**2
        inv_dy2=1/dy**2
        u_wall_count=v_wall_count=np.zeros((n,n))
        coord=(np.arange(n+2)-1)*dx
        X,Y=np.meshgrid(coord,coord)
        U=np.sin(X+dx/2)*np.cos(Y)
        V=-np.cos(X)*np.sin(Y+dy/2)
        P=rho/4*(np.cos(2*X)+np.cos(2*Y))
        for field in (U,V,P): periodic(field)
        fluid_center=np.ones((n,n),bool)
        a_e=a_w=np.full((n,n),inv_dx2)
        a_n=a_s=np.full((n,n),inv_dy2)
        pressure_denominator=a_e+a_w+a_n+a_s
        ii,jj=np.indices((n,n));red_cells=(ii+jj)%2==0;black_cells=~red_cells
        pressure_max_iterations=10000
        pressure_residual_tolerance=1e-8
        sor_omega=1.6
        core = CFDCore(
            dx=dx,
            dy=dy,
            dt=dt,
            dt_nu=dt_nu,
            inv_dx2=inv_dx2,
            inv_dy2=inv_dy2,
            u_wall_count=u_wall_count,
            v_wall_count=v_wall_count,
            a_e=a_e,
            a_w=a_w,
            a_n=a_n,
            a_s=a_s,
            pressure_denominator=pressure_denominator,
            fluid_center=fluid_center,
            red_cells=red_cells,
            black_cells=black_cells,
            pressure_max_iterations=pressure_max_iterations,
            pressure_residual_tolerance=pressure_residual_tolerance,
            sor_omega=sor_omega,
            apply_pressure_bc=apply_pressure_bc,
        )
        worst_div=0.0
        print(f'Running {n} x {n}, {steps} steps...',flush=True)
        for step in range(steps):
            us,vs=core.predict_velocity(U,V)
            periodic(us);periodic(vs)
            rhs=rho/dt*backward_divergence(us,vs)
            # Periodic divergence sums to zero; remove floating-point mean.
            rhs-=np.mean(rhs)
            P,it,res=solve_pressure(P,rhs)
            if not np.isfinite(res) or res>=pressure_residual_tolerance:
                raise RuntimeError(f'Pressure tolerance missed at step {step+1}: {res}')
            P-=np.mean(P[1:-1,1:-1]);periodic(P)
            U[1:-1,1:-1]=us[1:-1,1:-1]-dt/rho*(P[1:-1,2:]-P[1:-1,1:-1])/dx
            V[1:-1,1:-1]=vs[1:-1,1:-1]-dt/rho*(P[2:,1:-1]-P[1:-1,1:-1])/dy
            periodic(U);periodic(V)
            if not np.isfinite(U).all() or not np.isfinite(V).all():
                raise RuntimeError('Non-finite velocity')
            worst_div=max(worst_div,np.max(abs(backward_divergence(U,V))))
        decay=np.exp(-2*nu*end_time)
        ue=np.sin(X+dx/2)*np.cos(Y)*decay
        ve=-np.cos(X)*np.sin(Y+dy/2)*decay
        pe=rho/4*(np.cos(2*X)+np.cos(2*Y))*decay**2
        pe-=np.mean(pe[1:-1,1:-1])
        sl=np.s_[1:-1,1:-1]
        vel_error=np.sqrt(np.mean((U[sl]-ue[sl])**2+(V[sl]-ve[sl])**2))
        vel_norm=np.sqrt(np.mean(ue[sl]**2+ve[sl]**2))
        p_error=np.sqrt(np.mean((P[sl]-pe[sl])**2))
        records.append([n,dx,dt,vel_error,100*vel_error/vel_norm,p_error,worst_div])
        print(f'Velocity relative RMS error={100*vel_error/vel_norm:.6f}%; pressure RMS={p_error:.6e}; max divergence={worst_div:.3e}')
    data=np.asarray(records)
    for i in (1,2):
        print(f'Observed coupled refinement order: {np.log(data[i-1,3]/data[i,3])/np.log(2):.3f}')
    out=Path(__file__).resolve().parent/'vortex_upwind_verification_results';out.mkdir(exist_ok=True)
    np.savetxt(out/'errors.csv',data,delimiter=',',header='n,spacing,dt,velocity_rms,velocity_relative_percent,pressure_rms,worst_divergence',comments='')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    axs[0].loglog(data[:,1],data[:,3],'o-',label='Measured velocity error')
    axs[0].loglog(data[:,1],data[0,3]*data[:,1]/data[0,1],'--',label='First-order reference')
    axs[0].set(xlabel='Grid spacing',ylabel='Vector velocity RMS error',title='Grid and timestep refined together');axs[0].legend();axs[0].grid(alpha=.3)
    mid=n//2+1
    axs[1].plot(coord[1:-1]+dx/2,ue[mid,1:-1],label='Exact U')
    axs[1].plot(coord[1:-1]+dx/2,U[mid,1:-1],'--',label='Numerical U')
    axs[1].set(xlabel='x at U faces',ylabel='Horizontal velocity',title='Finest grid: horizontal section');axs[1].legend();axs[1].grid(alpha=.3)
    fig.tight_layout();fig.savefig(out/'vortex_verification.png',dpi=180)
    print('This test checks the periodic adapted kernel, not airfoil accuracy.')
    print('Both h and dt change; the order is not an isolated spatial-order estimate.')
    plt.show()
