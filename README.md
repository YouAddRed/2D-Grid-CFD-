# 2D-Grid-CFD-

# CFD
I made a 2D CFD that calculates lift drag coefficients and also other stuff like pressure difference.
Frozen on 09-23-2026. 

## Run the airfoil

Extract the whole ZIP and open the extracted folder in VS Code. In its terminal:
```powershell
python -m pip install -r requirements.txt
cd Airfoil
python Airfoil.py
```
Use the same interpreter for installation and compution. Airfoil.py requires the adjacent cfd_core.py and pressure_fallback.py. It runs 201 x 201, dt=0.0005, 10000 steps, angle zero, Re40, T=5 and displays/saves charts. NumPy 2 is required 

The optional Run_Grid_Check.py is the uploaded 301-grid runner. Run_401_Reconstructed.py selects the recorded 401-grid timestep-check settings, with a 400-step reporting interval if you would like to check those out.  Existing results are under results/; there is no need to repeat a long simulation to view them. Running scripts generates outputs separate from that evidence folder.

Optional component checks, from Airfoil/: python Verify_Pressure_Fallback.py; python Verify_Vortex.py; python Verify_Stepped_Wall.py. These tests have distinct scopes; see FREEZE_LOG.md.

## DFG benchmark
From DFG/DFG_Benchmark, run python DFG_Cylinder_Advection.py for the default cylinder case. DFG and airfoil use separate implementations.


Watching a rocket that works perfectly in theory, becomes a ball of fire in real life. To what extent should we put our truth in simulations or leave it to constant practical experimentation? What are the limitations that prevent the ethical from becoming displayed in the real world as they are seen on paper? 


#Summary


## 1. Scope of the project


This is a 2D CFD made by Jia Hong Ni. The CFD is a 2D grid 201x201 and 401x401 formatted and the grid is adjusted (the reason I chose 201 was because testing higher value grids took a lot of time off my not so good computer for compute). I first began with a cylinder and then continued into a common airfoil (NACA 0012). When I finished that section, I contacted Thu Nguyen, a CFD professional, and sent me over to test my model with TU Dortmund’s FEATFLOW website. Now the meat of this project: it contains the NACA 0012 simulation using my CFD code as the backbone in computation, a separate DFG simulation piece to benchmark, other test checks for other variables and different operations. While having the DFG to benchmark, I cannot say with certainty that it validates my work at all.


(URL here: https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html?)




## 2. Airfoil results


NACA 0012; Reynolds number 40; angle of attack 0 degrees; chord 0.4; final simulated time 5. Middle control volume supplies the force coefficients.


**on the left is using a 201x201 and the right is are values from running a 401x401g grid **


    | Quantity | Baseline | Refined case |


    | Grid | 201 × 201 | 401 × 401 |


    | Cells per chord | 32 | 64 |


    | Timestep | 0.0005 | 0.00025 |


    | Completed steps | 10,000 | 20,000 |


    | Drag coefficient, Cd | 0.79408263 | 0.77903349 |


    | Lift coefficient, Cl | Approximately zero | Approximately zero |


    | Relative Cd spread across control volumes | 0.4558% | 0.2203% |


    | Cd peak-to-peak / mean over T = 4–5 | 0.007647380% | 0.007386677% |


    | Maximum fluid divergence | 1.77679052e-14 | 3.55271368e-14 |


    | Pressure tolerance misses | 0/10,000 | 0/20,000 |


    ## Analysis
    The force settles as the reported discrete divergence becomes near zero and control-volume estimates are strong as they are within 1% or less. Refined-case drag is ~1.9% below baseline. Because both timestep and grid changed, the comparison does not account for spatial error. There is no matched external airfoil-force reference in the simulation.


    Sources within the ZIP: `FREEZE_LOG.md`, `Airfoil/results/201_dt0p0005/run.log`, and `Airfoil/results/401_dt0p00025/run.log`.


## 3. DFG cylinder benchmark


Steady two-dimensional flow around a cylinder at Re = 20. Reference: [TU Dortmund FEATFLOW, DFG 2D-1](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html?). The published page attributes the benchmark to Schäfer and Turek (1996), and its reference values to Nabh (1998).


    | Case | Cd | Cl | Front-to-rear pressure difference |


    | Published reference | 5.579535234 | 0.010618948 | 0.117520167 |


    | 440 × 82, dt = 0.002 | 5.612441623 | 0.012020560 | 0.111287834 |


    | 440 × 82, dt = 0.001 | 5.612441623 | 0.012020563 | 0.111287834 |


    | 660 × 123, dt = 0.002 | 5.635699205 | 0.011763987 | 0.115172529 |


What I use for comparison was the percentage relative difference = 100 × (computed − reference) / reference.


    | Case | Cd difference | Cl difference | Pressure-difference deviation |


    | 440 × 82, dt = 0.002 | +0.590% | +13.199% | −5.303% |


    | 660 × 123, dt = 0.002 | +1.007% | +10.783% | −1.998% |


**Interpretation:** The smaller timestep barely changed the tested steady solution. The finer grid improves lift and pressure agreement but worsens drag from the published value. These deviations are meant to be comparisons and not a result the CFD is trying to compute exactly. Although there is roughly a 5% difference, my value differs from the published value by 5.6163971E-2. Looking into revisions, this might have been a result of different ways of calculating the pressure.  


## 4. Component verification evidence
This section is basically checking different parts being verified and tested separately 
| Test | Evidence available | Supported conclusion | Limit |


| Discrete pressure recovery | Frozen historical TEST_RESULTS.txt: 
recovery error 1.105782e-13; stencil residual 1.236913e-10 | The tested discrete pressure problem is recovered accurately | Does not verify the entire flow solution |
	
	Discrete Pressure Recovery is basically whether or not our pressure solver incorporates a known value and uses it. Our values for recovery error and residual are within the tenth billionths place value which implies our value as very precise.
 
| Periodic vortex | Verify_Vortex.py is included; earlier conversation contains run output, but no separate result log is included in this freeze | A verification mechanism for the adapted periodic kernel is preserved | Do not assign earlier numerical results to this exact script without confirming provenance |


	I decided to use a taylor green vortex to test whether the numerical method would be able to handle if flow were to change. This also serves as a buffer for flow as this test helps avoid the airfoil or solid walls to complicate the entire test. 


| Smooth stepped-wall diffusion | Verify_Stepped_Wall.py is included; freeze log records reported near-second-order behavior | Evidence addresses smooth diffusion and mixed wall stencils | Does not establish coupled Navier–Stokes corner accuracy or airfoil forces |


	Near the boundaries of the CFD, there are set boundaries that I established to make sure the function for diffusion would behave correctly near the stair-stepped approximations of curves from my cartesian grid cell. 




Velocity RMS errors of 13.330141%, 3.464691%, 0.874535%, and 0.219159% at 16, 32, 64, and 128 intervals. The refinement orders were 1.944, 1.986, and 1.997. These are conversation-reported results, not a newly executed test or an archived output log tied to the frozen script.


	RMS error is just the overall differences across the sample instead of just error at one point (since computing that is very heavy). My idea of doubling the resolution was very beneficial in cutting the error by approximately 4x each time. The refinement orders come p = ln(E coarse / E fine) all divided by ln 2. The refinement isn't as important as it describes how the test improves with refinement and eventually there's no point in continuing as the cons outweigh the pros. 


## 5. Everything is relative
Everything below is RELATIVE to the extracted `CFD_Portfolio` directory.


### Figure 1 — DFG benchmark comparison


File: `DFG/DFG_Benchmark/comparison.png`.


Caption: Drag, lift, and front-to-rear pressure difference for the DFG Re20 cylinder. The 440 × 82 with dt= 0.001 changes the results compared to 440 × 82 with dt of 0.002 by so little that we can consider them the same. Refining to 660 × 123 improves lift and pressure agreement but deviates our drag coefficient away from the published value. Dashed lines show the published reference values. The comparison retains all three quantities rather than selecting only the closest result.


Visual Graphs: Include the full grid dimensions in the caption because the figure's horizontal labels show only 440 or 660, a graph showing how front-to-rear pressure climbs and the difference between them. 


### Figure 2 — Airfoil flow fields (selected)


File: `Airfoil/results/401_dt0p00025/01_flow_fields.png` 


Caption: Computed velocity and pressure fields around NACA 0012 at Re40 and zero angle of attack on a 401 × 401 Cartesian grid, at T = 5. The visualization shows the resolved flow and wake around the numerical solid mask. The point of this png is to set the streamline visualization as an application to how the field appears and does not have any aerodynamic accuracy.


### Figure 3 — Airfoil force settling (selected)


File: `Airfoil/results/401_dt0p00025/forces.png`.


Caption: Middle-control-volume force history for the 401 × 401 airfoil case. Drag approaches Cd = 0.77903349 while the unsteady streamwise force contribution decays. Over T = 4–5, drag peak-to-peak variation is 0.007386677% of its mean. Lift variations near 1e-15 are roundoff-level differences, not resolved physical oscillations.


Presentation note: The right-hand panels magnify very small values. State their scale prominently in the report. A steady coefficient is not proof of absolute accuracy.


### Figure 4 — Airfoil geometry and wake (selected)


File: `Airfoil/results/401_dt0p00025/03_wake_and_geometry.png` (PDF also available).


Caption: Wake profiles and the Cartesian representation of the airfoil at 64 cells. The mask reveals the staircase geometry that remains despite refinement. At zero angle of attack the final stream and historical horizontal velocity directions coincide and this equivalence does not appear with other angles of attack.


### Supporting figures and exclusions


- `04_convergence_history.png`: 2 charts that display mass-error at 1e-16 is a logarithmic axis thus it cannot show zero so we plot a very small value instead. SOR iterations is the method for solving pressure. I use a sparse pressure solver, but pressure will still be calculated after the SOR. 
- `02_pressure_loading.png`: These are the pressures above and below the airfoil. The code samples pressure at nearby fluid grid locations to describe the near surface pressure estimates. The difference of 10e-15 is near zero but the graph makes it look like a massive difference but we can assume it to be the same.


## 6. Report outline and claim boundaries


Suggested order: motivation and scope → numerical methods and implementation differences → component checks → DFG benchmark table and Figure 1 → airfoil application and Figures 2–4 → limitations and lessons → reproduction instructions and acknowledgments.


Reminder that this is a student made CFD project with component verification, a separate cylinder benchmark comparison, and a low-Reynolds-number airfoil application with recorded numerical sensitivity. MY CFD uses a Reynolds number of 40 while real qualified CFDs ,for airfoil specifically, are normally in the tens of millions and given that massive difference, none of my results are quantifiable to be used in a professional or published research setting. 



