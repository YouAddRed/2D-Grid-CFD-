from cfd_core import CFDCore
from pressure_fallback import AirfoilPressureFallback
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Basic Settings
nx = 201
ny = 201

dt = 5.0e-4
num_steps = 10000

steady_check_interval = 200
steady_velocity_tolerance = 1.0e-4
steady_required_checks = 3
minimum_steady_step = 2000

STOP_AT_STEADY_STATE = False

domain_x_min = 0.0
domain_x_max = 2.5

domain_y_min = 0.0
domain_y_max = 2.5

x = np.linspace(
    domain_x_min,
    domain_x_max,
    nx
)

y = np.linspace(
    domain_y_min,
    domain_y_max,
    ny
)

X, Y = np.meshgrid(
    x,
    y
)

dx = x[1] - x[0]
dy = y[1] - y[0]

rho = 1.0
nu = 0.01

SHOW_PLOTS = True


# Airfoil Geometry

base_chord = 0.20
scale_factor = 2.0 

chord = base_chord * scale_factor

x_le = 1.05
y_center = 1.25

t = 0.12

angle_of_attack = 0.0

alpha = np.radians(angle_of_attack)


# Inlet Velocity

U_inf_input = 1.0

U_inlet = U_inf_input * np.cos(alpha)
V_inlet = U_inf_input * np.sin(alpha)

U_inf = np.sqrt(
    U_inlet**2 +
    V_inlet**2
)


# Naca 0012 Geometry

num_airfoil_points = 300

x_airfoil = np.linspace(
    0.0,
    chord,
    num_airfoil_points
)

x_bar = x_airfoil / chord

yt = 5.0 * t * chord * (
    0.2969 * np.sqrt(x_bar)
    - 0.1260 * x_bar
    - 0.3516 * x_bar**2
    + 0.2843 * x_bar**3
    - 0.1015 * x_bar**4
)

x_surface = x_le + x_airfoil

y_upper = y_center + yt
y_lower = y_center - yt


# Solid Mask

solid = np.zeros_like(
    X,
    dtype=bool
)

inside_x = (
    (X >= x_le)
    &
    (X <= x_le + chord)
)

upper_grid = np.interp(
    X,
    x_surface,
    y_upper
)

lower_grid = np.interp(
    X,
    x_surface,
    y_lower
)

solid = (
    inside_x
    &
    (Y >= lower_grid)
    &
    (Y <= upper_grid)
)

fluid = ~solid


# Open Face Masks

u_face_open = (
    fluid[:, :-1]
    &
    fluid[:, 1:]
)

v_face_open = (
    fluid[:-1, :]
    &
    fluid[1:, :]
)


# Straight-wall correction and mixed-corner audit
u_active = u_face_open[1:-1, 1:]
v_active = v_face_open[1:, 1:-1]
u_wall_count = ((u_active & solid[:-2, 1:-1] & solid[:-2, 2:]).astype(float)
                + (u_active & solid[2:, 1:-1] & solid[2:, 2:]).astype(float))
v_wall_count = ((v_active & solid[1:-1, :-2] & solid[2:, :-2]).astype(float)
                + (v_active & solid[1:-1, 2:] & solid[2:, 2:]).astype(float))
u_mixed_south = u_active & (solid[:-2, 1:-1] ^ solid[:-2, 2:])
u_mixed_north = u_active & (solid[2:, 1:-1] ^ solid[2:, 2:])
v_mixed_west = v_active & (solid[1:-1, :-2] ^ solid[2:, :-2])
v_mixed_east = v_active & (solid[1:-1, 2:] ^ solid[2:, 2:])
print("Corner audit: mixed neighbor configurations (not unique corners)")
for label, mask in [("U south", u_mixed_south), ("U north", u_mixed_north),
                    ("V west", v_mixed_west), ("V east", v_mixed_east)]:
    print(f"{label}: {np.count_nonzero(mask)}")
print("Mixed configurations retain the existing wall treatment.")
# Save physical locations so corner treatment can be reviewed explicitly.
corner_rows = []
for component, direction, mask in [("U", "south", u_mixed_south),
        ("U", "north", u_mixed_north), ("V", "west", v_mixed_west),
        ("V", "east", v_mixed_east)]:
    for i, j in zip(*np.nonzero(mask)):
        corner_rows.append([component, direction,
            x[j+1] + (dx/2 if component == "U" else 0),
            y[i+1] + (dy/2 if component == "V" else 0)])
import csv
corner_path = Path(__file__).resolve().parent / "corner_stencil_audit.csv"
with corner_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["component", "neighbor_direction", "face_x", "face_y"])
    writer.writerows(corner_rows)


# Pressure Solver Settings

pressure_max_iterations = 1500

sor_omega = 1.6

target_divergence = 1.0e-5

pressure_residual_tolerance = (
    rho / dt
) * target_divergence


# Velocity And Pressure Fields

U = np.zeros_like(X)
V = np.zeros_like(X)
P = np.zeros_like(X)


# Boundary Conditions

def apply_velocity_bc(U_field, V_field):

    U_field[:, 0] = U_inlet
    V_field[:, 0] = V_inlet

    U_field[:, -1] = U_field[:, -2]
    V_field[:, -1] = V_field[:, -2]

    # U is stored at cell-row locations. V is stored on the
    # horizontal face above each cell, so its top boundary face
    # is row -2 rather than row -1.
    U_field[0, :] = U_inlet
    U_field[-1, :] = U_inlet

    V_field[0, :] = V_inlet
    V_field[-2, :] = V_inlet
    V_field[-1, :] = V_inlet

    U_faces = U_field[:, :-1]
    V_faces = V_field[:-1, :]

    U_faces[~u_face_open] = 0.0
    V_faces[~v_face_open] = 0.0

    # Reapply the external far-field faces after face masking.
    U_field[0, :] = U_inlet
    U_field[-1, :] = U_inlet
    V_field[0, :] = V_inlet
    V_field[-2, :] = V_inlet
    V_field[-1, :] = V_inlet

def apply_pressure_bc(P_field):

    P_field[:, 0] = P_field[:, 1]

    P_field[0, :] = P_field[1, :]

    P_field[-1, :] = P_field[-2, :]

    P_field[:, -1] = 0.0

def enforce_zero_angle_symmetry(U_field=None, V_field=None, P_field=None):
    # Symmetry audit: leave the calculated fields unchanged.
    return


# Divergence



# Mass Flow

def calculate_fluxes(U_field):

    inlet_open = fluid[1:-1, 0]

    outlet_open = fluid[1:-1, -2]

    U_inlet_field = U_field[1:-1, 0]

    U_outlet_field = U_field[1:-1, -2]

    Q_in = (
        np.sum(
            U_inlet_field[inlet_open]
        )
        * dy
    )

    Q_out = (
        np.sum(
            U_outlet_field[outlet_open]
        )
        * dy
    )

    mass_error = Q_out - Q_in

    relative_mass_error = (
        mass_error
        /
        max(
            abs(Q_in),
            1.0e-14
        )
    )

    return (
        Q_in,
        Q_out,
        mass_error,
        relative_mass_error
    )


# Pressure Poisson Coefficients

fluid_center = fluid[1:-1, 1:-1]

fluid_e = fluid[1:-1, 2:]
fluid_w = fluid[1:-1, :-2]

fluid_n = fluid[2:, 1:-1]
fluid_s = fluid[:-2, 1:-1]

inv_dx2 = 1.0 / dx**2
inv_dy2 = 1.0 / dy**2

a_e = (
    fluid_e.astype(float)
    *
    inv_dx2
)

a_w = (
    fluid_w.astype(float)
    *
    inv_dx2
)

a_n = (
    fluid_n.astype(float)
    *
    inv_dy2
)

a_s = (
    fluid_s.astype(float)
    *
    inv_dy2
)

pressure_denominator = (
    a_e
    +
    a_w
    +
    a_n
    +
    a_s
)


row_indices, col_indices = np.indices(
    fluid_center.shape
)

red_cells = (
    (
        (row_indices + col_indices)
        % 2
        == 0
    )
    &
    fluid_center
)

black_cells = (
    (
        (row_indices + col_indices)
        % 2
        == 1
    )
    &
    fluid_center
)


# Pressure Candidate



# Pressure Residual



# Pressure Solver



# Initial Conditions

apply_velocity_bc(
    U,
    V
)

apply_pressure_bc(
    P
)


# Data Collection

profile_times = sorted(set(range(steady_check_interval, num_steps + 1,
                                 steady_check_interval)) | {num_steps})

profiles = {}

time_history = []

max_velocity_history = []

divergence_history = []

pressure_iterations_history = []

mass_error_history = []

relative_mass_error_history = []

last_pressure_iterations = 0

last_pressure_residual = 0.0


# Precomputed Constants

dt_nu = nu * dt

dt_over_rho = dt / rho

rho_over_dt = rho / dt

U_before_final_step = None
V_before_final_step = None

def run_channel_wall_test():
    print("\nStraight-wall channel component test:")
    print(
        "Cells | Zero-solid-row error | "
        "Face-aligned ghost error"
    )

    for channel_cells in (16, 32, 64):
        channel_dy = 1.0 / channel_cells

        channel_y = (
            np.arange(channel_cells) + 0.5
        ) * channel_dy

        # Exact solution of -u'' = 8 with u(0) = u(1) = 0.
        exact_velocity = (
            4.0 * channel_y * (1.0 - channel_y)
        )

        # Zero velocity at neighboring solid-row locations.
        channel_matrix = (
            2.0 * np.eye(channel_cells)
            - np.eye(channel_cells, k=1)
            - np.eye(channel_cells, k=-1)
        )

        channel_rhs = np.full(
            channel_cells,
            8.0 * channel_dy**2
        )

        zero_solid_velocity = np.linalg.solve(
            channel_matrix,
            channel_rhs
        )

        # Alternative: zero velocity at the intervening wall face.
        # The ghost value is the negative of the fluid value.
        face_matrix = channel_matrix.copy()
        face_matrix[0, 0] = 3.0
        face_matrix[-1, -1] = 3.0

        face_aligned_velocity = np.linalg.solve(
            face_matrix,
            channel_rhs
        )

        # Normalize by the exact peak velocity, which is 1.
        zero_solid_error = np.max(
            np.abs(zero_solid_velocity - exact_velocity)
        )

        face_aligned_error = np.max(
            np.abs(face_aligned_velocity - exact_velocity)
        )

        print(
            f"{channel_cells:5d} | "
            f"{100.0 * zero_solid_error:19.4f}% | "
            f"{100.0 * face_aligned_error:23.4f}%"
        )

    print(
        "Errors are relative to the exact channel peak speed."
    )
    print(
        "This is a straight-wall diffusion component test, "
        "not validation of airfoil lift or drag.\n"
    )


run_channel_wall_test()

def sample_control_volume_forces(U, V, P, U_before_final_step, V_before_final_step):
    # Same calculation as final reporting, evaluated on the current step.
    q_inf = 0.5 * rho * U_inf**2
    mu = rho * nu


    # Convert the staggered face velocities to the pressure-cell
    # centers used by the control-volume integration.

    def velocities_at_cell_centers(U_face, V_face):

        U_cell = U_face.copy()
        V_cell = V_face.copy()

        U_cell[:, 1:-1] = 0.5 * (
            U_face[:, 1:-1]
            +
            U_face[:, :-2]
        )

        V_cell[1:-1, :] = 0.5 * (
            V_face[1:-1, :]
            +
            V_face[:-2, :]
        )

        V_cell[-1, :] = V_face[-2, :]

        return U_cell, V_cell


    U_force, V_force = velocities_at_cell_centers(
        U,
        V
    )

    (
        U_before_final_force,
        V_before_final_force
    ) = velocities_at_cell_centers(
        U_before_final_step,
        V_before_final_step
    )

    control_volumes = [
        {
            "name": "Inner",
            "x_left": x_le - 0.25 * chord,
            "x_right": x_le + 1.25 * chord,
            "y_bottom": y_center - 0.625 * chord,
            "y_top": y_center + 0.625 * chord
        },

        {
            "name": "Middle",
            "x_left": x_le - 0.375 * chord,
            "x_right": x_le + 1.375 * chord,
            "y_bottom": y_center - 0.75 * chord,
            "y_top": y_center + 0.75 * chord
        },

        {
            "name": "Outer",
            "x_left": x_le - 0.50 * chord,
            "x_right": x_le + 1.50 * chord,
            "y_bottom": y_center - 0.875 * chord,
            "y_top": y_center + 0.875 * chord
        }
    ]

    airfoil_x_min = x_le
    airfoil_x_max = x_le + chord
    airfoil_y_min = np.min(y_lower)
    airfoil_y_max = np.max(y_upper)

    if airfoil_x_min <= domain_x_min or airfoil_x_max >= domain_x_max:
        raise ValueError(
            "The scaled airfoil does not fit inside the x-domain. "
            "Increase the domain size or move x_le."
        )

    if airfoil_y_min <= domain_y_min or airfoil_y_max >= domain_y_max:
        raise ValueError(
            "The scaled airfoil does not fit inside the y-domain. "
            "Increase the domain size or move y_center."
        )

    for cv in control_volumes:
        if (
            cv["x_left"] <= domain_x_min
            or cv["x_right"] >= domain_x_max
            or cv["y_bottom"] <= domain_y_min
            or cv["y_top"] >= domain_y_max
        ):
            raise ValueError(
                f'{cv["name"]} control volume does not fit inside the domain. '
                "Increase the domain size, reduce scale_factor, or reposition the airfoil."
            )

    dU_dx_cv = np.zeros_like(U_force)
    dU_dy_cv = np.zeros_like(U_force)
    dV_dx_cv = np.zeros_like(V_force)
    dV_dy_cv = np.zeros_like(V_force)

    dU_dx_cv[:, 1:-1] = (
        U_force[:, 2:]
        - U_force[:, :-2]
    ) / (2.0 * dx)

    dU_dy_cv[1:-1, :] = (
        U_force[2:, :]
        - U_force[:-2, :]
    ) / (2.0 * dy)

    dV_dx_cv[:, 1:-1] = (
        V_force[:, 2:]
        - V_force[:, :-2]
    ) / (2.0 * dx)

    dV_dy_cv[1:-1, :] = (
        V_force[2:, :]
        - V_force[:-2, :]
    ) / (2.0 * dy)

    tau_xx = 2.0 * mu * dU_dx_cv
    tau_yy = 2.0 * mu * dV_dy_cv
    tau_xy = mu * (dU_dy_cv + dV_dx_cv)

    dU_dt_final = (
        U_force - U_before_final_force
    ) / dt

    dV_dt_final = (
        V_force - V_before_final_force
    ) / dt

    def calculate_control_volume_force(
        cv_x_left_target,
        cv_x_right_target,
        cv_y_bottom_target,
        cv_y_top_target
    ):

        cv_j_left = np.argmin(
            np.abs(x - cv_x_left_target)
        )

        cv_j_right = np.argmin(
            np.abs(x - cv_x_right_target)
        )

        cv_i_bottom = np.argmin(
            np.abs(y - cv_y_bottom_target)
        )

        cv_i_top = np.argmin(
            np.abs(y - cv_y_top_target)
        )

        if cv_j_left >= cv_j_right:
            raise ValueError(
                "Control-volume x limits are invalid."
            )

        if cv_i_bottom >= cv_i_top:
            raise ValueError(
                "Control-volume y limits are invalid."
            )

        cv_x_left = x[cv_j_left]
        cv_x_right = x[cv_j_right]
        cv_y_bottom = y[cv_i_bottom]
        cv_y_top = y[cv_i_top]

        solid_on_left = np.any(
            solid[
                cv_i_bottom:cv_i_top + 1,
                cv_j_left
            ]
        )

        solid_on_right = np.any(
            solid[
                cv_i_bottom:cv_i_top + 1,
                cv_j_right
            ]
        )

        solid_on_bottom = np.any(
            solid[
                cv_i_bottom,
                cv_j_left:cv_j_right + 1
            ]
        )

        solid_on_top = np.any(
            solid[
                cv_i_top,
                cv_j_left:cv_j_right + 1
            ]
        )

        if (
            solid_on_left
            or solid_on_right
            or solid_on_bottom
            or solid_on_top
        ):
            raise ValueError(
                "Control volume intersects the solid airfoil."
            )

        y_cv = y[cv_i_bottom:cv_i_top + 1]
        x_cv = x[cv_j_left:cv_j_right + 1]

        U_left = U_force[
            cv_i_bottom:cv_i_top + 1,
            cv_j_left
        ]
        V_left = V_force[
            cv_i_bottom:cv_i_top + 1,
            cv_j_left
        ]
        P_left = P[
            cv_i_bottom:cv_i_top + 1,
            cv_j_left
        ]

        U_right = U_force[
            cv_i_bottom:cv_i_top + 1,
            cv_j_right
        ]
        V_right = V_force[
            cv_i_bottom:cv_i_top + 1,
            cv_j_right
        ]
        P_right = P[
            cv_i_bottom:cv_i_top + 1,
            cv_j_right
        ]

        U_bottom = U_force[
            cv_i_bottom,
            cv_j_left:cv_j_right + 1
        ]
        V_bottom = V_force[
            cv_i_bottom,
            cv_j_left:cv_j_right + 1
        ]
        P_bottom = P[
            cv_i_bottom,
            cv_j_left:cv_j_right + 1
        ]

        U_top = U_force[
            cv_i_top,
            cv_j_left:cv_j_right + 1
        ]
        V_top = V_force[
            cv_i_top,
            cv_j_left:cv_j_right + 1
        ]
        P_top = P[
            cv_i_top,
            cv_j_left:cv_j_right + 1
        ]

        Fx_pressure_cv = (
            np.trapezoid(P_left, y_cv)
            - np.trapezoid(P_right, y_cv)
        )

        Fy_pressure_cv = (
            np.trapezoid(P_bottom, x_cv)
            - np.trapezoid(P_top, x_cv)
        )

        Fx_viscous_cv = (
            np.trapezoid(
                -tau_xx[
                    cv_i_bottom:cv_i_top + 1,
                    cv_j_left
                ],
                y_cv
            )
            + np.trapezoid(
                tau_xx[
                    cv_i_bottom:cv_i_top + 1,
                    cv_j_right
                ],
                y_cv
            )
            + np.trapezoid(
                -tau_xy[
                    cv_i_bottom,
                    cv_j_left:cv_j_right + 1
                ],
                x_cv
            )
            + np.trapezoid(
                tau_xy[
                    cv_i_top,
                    cv_j_left:cv_j_right + 1
                ],
                x_cv
            )
        )

        Fy_viscous_cv = (
            np.trapezoid(
                -tau_xy[
                    cv_i_bottom:cv_i_top + 1,
                    cv_j_left
                ],
                y_cv
            )
            + np.trapezoid(
                tau_xy[
                    cv_i_bottom:cv_i_top + 1,
                    cv_j_right
                ],
                y_cv
            )
            + np.trapezoid(
                -tau_yy[
                    cv_i_bottom,
                    cv_j_left:cv_j_right + 1
                ],
                x_cv
            )
            + np.trapezoid(
                tau_yy[
                    cv_i_top,
                    cv_j_left:cv_j_right + 1
                ],
                x_cv
            )
        )

        Fx_momentum_cv = (
            np.trapezoid(
                rho * U_left**2,
                y_cv
            )
            - np.trapezoid(
                rho * U_right**2,
                y_cv
            )
            + np.trapezoid(
                rho * U_bottom * V_bottom,
                x_cv
            )
            - np.trapezoid(
                rho * U_top * V_top,
                x_cv
            )
        )

        Fy_momentum_cv = (
            np.trapezoid(
                rho * U_left * V_left,
                y_cv
            )
            - np.trapezoid(
                rho * U_right * V_right,
                y_cv
            )
            + np.trapezoid(
                rho * V_bottom**2,
                x_cv
            )
            - np.trapezoid(
                rho * V_top**2,
                x_cv
            )
        )

        dU_dt_inside_cv = dU_dt_final[
            cv_i_bottom:cv_i_top + 1,
            cv_j_left:cv_j_right + 1
        ]

        dV_dt_inside_cv = dV_dt_final[
            cv_i_bottom:cv_i_top + 1,
            cv_j_left:cv_j_right + 1
        ]

        integrated_dU_dt_x = np.trapezoid(
            dU_dt_inside_cv,
            x_cv,
            axis=1
        )

        integrated_dV_dt_x = np.trapezoid(
            dV_dt_inside_cv,
            x_cv,
            axis=1
        )

        Fx_unsteady_cv = (
            -rho
            * np.trapezoid(
                integrated_dU_dt_x,
                y_cv
            )
        )

        Fy_unsteady_cv = (
            -rho
            * np.trapezoid(
                integrated_dV_dt_x,
                y_cv
            )
        )

        Fx_total_cv = (
            Fx_pressure_cv
            + Fx_viscous_cv
            + Fx_momentum_cv
            + Fx_unsteady_cv
        )

        Fy_total_cv = (
            Fy_pressure_cv
            + Fy_viscous_cv
            + Fy_momentum_cv
            + Fy_unsteady_cv
        )

        drag_cv = (
            Fx_total_cv * np.cos(alpha)
            + Fy_total_cv * np.sin(alpha)
        )

        lift_cv = (
            -Fx_total_cv * np.sin(alpha)
            + Fy_total_cv * np.cos(alpha)
        )

        Cd_cv = drag_cv / (q_inf * chord)
        Cl_cv = lift_cv / (q_inf * chord)

        return {
            "x_left": cv_x_left,
            "x_right": cv_x_right,
            "y_bottom": cv_y_bottom,
            "y_top": cv_y_top,
            "Fx_pressure": Fx_pressure_cv,
            "Fy_pressure": Fy_pressure_cv,
            "Fx_viscous": Fx_viscous_cv,
            "Fy_viscous": Fy_viscous_cv,
            "Fx_momentum": Fx_momentum_cv,
            "Fy_momentum": Fy_momentum_cv,
            "Fx_unsteady": Fx_unsteady_cv,
            "Fy_unsteady": Fy_unsteady_cv,
            "Fx_total": Fx_total_cv,
            "Fy_total": Fy_total_cv,
            "drag": drag_cv,
            "lift": lift_cv,
            "Cd": Cd_cv,
            "Cl": Cl_cv
        }

    cv_results = []

    for cv in control_volumes:
        result = calculate_control_volume_force(
            cv["x_left"],
            cv["x_right"],
            cv["y_bottom"],
            cv["y_top"]
        )

        result["name"] = cv["name"]
        cv_results.append(result)

    Cd_values = np.array([
        result["Cd"]
        for result in cv_results
    ])

    Cl_values = np.array([
        result["Cl"]
        for result in cv_results
    ])

    Cd_mean = np.mean(Cd_values)
    Cl_mean = np.mean(Cl_values)

    Cd_spread = (
        np.max(Cd_values)
        - np.min(Cd_values)
    )

    Cl_spread = (
        np.max(Cl_values)
        - np.min(Cl_values)
    )

    Cd_relative_spread = (
        Cd_spread
        / max(abs(Cd_mean), 1.0e-14)
    )

    reference_result = next(
        result
        for result in cv_results
        if result["name"] == "Middle"
    )

    Fx_total = reference_result["Fx_total"]
    Fy_total = reference_result["Fy_total"]
    drag = reference_result["drag"]
    lift = reference_result["lift"]
    Cd = reference_result["Cd"]
    Cl = reference_result["Cl"]
    return reference_result, Cd_relative_spread

force_history = []

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

def backward_divergence(U_field, V_field):
    return core.backward_divergence(U_field, V_field)

pressure_fallback = AirfoilPressureFallback(core)

def solve_pressure(P_field, rhs):
    if pressure_fallback.factor is not None:
        P_field, residual = pressure_fallback.solve(P_field, rhs)
        return P_field, 0, residual
    P_field, iterations, residual = core.solve_pressure(P_field, rhs)
    if not np.isfinite(residual) or residual >= pressure_residual_tolerance:
        P_field, residual = pressure_fallback.solve(P_field, rhs)
        if pressure_fallback.calls <= 3:
            print(f"Pressure fallback accepted: residual={residual:.3e}", flush=True)
    return P_field, iterations, residual


# Main Cfd Loop

previous_check_max_speed = None
steady_check_count = 0
completed_steps = 0
steady_state_reached = False

pressure_failed_steps = 0
pressure_total_iterations = 0
pressure_worst_residual = 0.0

for step in range(num_steps):

    U_before_final_step = U.copy()
    V_before_final_step = V.copy()

    U_star, V_star = core.predict_velocity(U, V)

    # Apply boundary conditions

    apply_velocity_bc(
        U_star,
        V_star
    )

    enforce_zero_angle_symmetry(
        U_field=U_star,
        V_field=V_star
    )


    # Divergence before pressure correction

    divergence_before = backward_divergence(
        U_star,
        V_star
    )


    # Pressure Poisson RHS

    rhs = (
        rho_over_dt
        *
        divergence_before
    )


    # Solve pressure

    (
        P,
        last_pressure_iterations,
        last_pressure_residual
    ) = solve_pressure(
        P,
        rhs
    )

    if not np.isfinite(last_pressure_residual):
        raise RuntimeError(f"Non-finite pressure residual at step {step + 1}")
    pressure_total_iterations += last_pressure_iterations
    pressure_worst_residual = max(pressure_worst_residual, last_pressure_residual)
    if last_pressure_residual >= pressure_residual_tolerance:
        pressure_failed_steps += 1
        if pressure_failed_steps <= 3:
            print(f"Pressure tolerance missed at step {step + 1}: "
                  f"residual = {last_pressure_residual:.3e}, "
                  f"target = {pressure_residual_tolerance:.3e}")

    enforce_zero_angle_symmetry(
        P_field=P
    )


    # Pressure gradients

    raw_dP_dx = (
        P[1:-1, 2:]
        -
        P[1:-1, 1:-1]
    ) / dx

    raw_dP_dy = (
        P[2:, 1:-1]
        -
        P[1:-1, 1:-1]
    ) / dy


    # Valid pressure corrections

    valid_u_correction = (
        fluid_center
        &
        fluid_e
    )

    valid_v_correction = (
        fluid_center
        &
        fluid_n
    )


    dP_dx = np.zeros_like(
        raw_dP_dx
    )

    dP_dy = np.zeros_like(
        raw_dP_dy
    )

    dP_dx[valid_u_correction] = (
        raw_dP_dx[
            valid_u_correction
        ]
    )

    dP_dy[valid_v_correction] = (
        raw_dP_dy[
            valid_v_correction
        ]
    )


    # Pressure correction

    U[1:-1, 1:-1] = (
        U_star[1:-1, 1:-1]
        -
        dt_over_rho * dP_dx
    )

    V[1:-1, 1:-1] = (
        V_star[1:-1, 1:-1]
        -
        dt_over_rho * dP_dy
    )


    # Final boundary conditions

    apply_velocity_bc(
        U,
        V
    )

    enforce_zero_angle_symmetry(
        U_field=U,
        V_field=V
    )

    completed_steps = step + 1
    if completed_steps % steady_check_interval == 0 or completed_steps == num_steps:
        force_sample, force_spread = sample_control_volume_forces(
            U, V, P, U_before_final_step, V_before_final_step)
        force_history.append([
            completed_steps * dt, force_sample["Cd"], force_sample["Cl"],
            force_sample["Fx_unsteady"], force_sample["Fy_unsteady"], force_spread])
        print(f"Forces at T={completed_steps * dt:.4f}: "
              f"Cd={force_sample['Cd']:.8f}, Cl={force_sample['Cl']:.8f}", flush=True)


    if completed_steps <= 200 and completed_steps % 10 == 0:
        print(
            f"Startup step {completed_steps}/200 | "
            f"SOR iterations = {last_pressure_iterations} | "
            f"residual = {last_pressure_residual:.3e}",
            flush=True
        )
    # Data Collection

    if completed_steps in profile_times:

        divergence_after = backward_divergence(
            U,
            V
        )

        (
            Q_in,
            Q_out,
            mass_error,
            relative_mass_error
        ) = calculate_fluxes(
            U
        )

        profiles[
            completed_steps
        ] = U.copy()

        time_history.append(
            completed_steps * dt
        )

        max_velocity_history.append(
            np.max(np.hypot(
                0.5 * (U[1:-1, 1:-1] + U[1:-1, :-2]),
                0.5 * (V[1:-1, 1:-1] + V[:-2, 1:-1])
            )[fluid_center])
        )

        divergence_history.append(
            np.max(
                np.abs(
                    divergence_after
                )
            )
        )

        pressure_iterations_history.append(
            last_pressure_iterations
        )

        mass_error_history.append(
            mass_error
        )

        relative_mass_error_history.append(
            relative_mass_error
        )


    # Progress Output And Steady-State Check

    if completed_steps % steady_check_interval == 0:

        (
            Q_in,
            Q_out,
            mass_error,
            relative_mass_error
        ) = calculate_fluxes(
            U
        )

        current_max_speed = np.max(
            np.sqrt(U**2 + V**2)
        )

        print(
            f"Step {completed_steps:5d} | "
            f"max speed = "
            f"{current_max_speed:.5f} | "
            f"Qin = {Q_in:.6f} | "
            f"Qout = {Q_out:.6f} | "
            f"mass error = "
            f"{relative_mass_error:.3e} | "
            f"SOR iterations = "
            f"{last_pressure_iterations:3d} | "
            f"Poisson residual = "
            f"{last_pressure_residual:.3e}"
        )

        if previous_check_max_speed is not None:

            speed_change = abs(
                current_max_speed
                -
                previous_check_max_speed
            )

            if completed_steps >= minimum_steady_step:

                if speed_change < steady_velocity_tolerance:
                    steady_check_count += 1
                else:
                    steady_check_count = 0

                print(
                    f"Steady checks = "
                    f"{steady_check_count}/"
                    f"{steady_required_checks} | "
                    f"max-speed change = "
                    f"{speed_change:.3e}"
                )

                if (
                    STOP_AT_STEADY_STATE
                    and
                    steady_check_count >= steady_required_checks
                ):
                    steady_state_reached = True
                    print()
                    print(
                        f"Steady-state criterion reached "
                        f"at step {completed_steps}."
                    )
                    break

        previous_check_max_speed = current_max_speed


print("\nPressure convergence audit:")
print(f"Sparse pressure solves: {pressure_fallback.calls}")
print("Zero SOR iterations means the cached sparse solver was used.")
print(f"Steps missing tolerance: {pressure_failed_steps}/{completed_steps}")
print(f"Average SOR iterations: {pressure_total_iterations / max(completed_steps, 1):.1f}")
print(f"Worst pressure residual: {pressure_worst_residual:.6e}")

# Full-domain boundary flux audit
# These indices bound the interior cells used by
# backward_divergence(). Positive flux means outward flow.

flux_left = (
    -np.sum(U[1:-1, 0]) * dy
)

flux_right = (
    np.sum(U[1:-1, -2]) * dy
)

flux_bottom = (
    -np.sum(V[0, 1:-1]) * dx
)

flux_top = (
    np.sum(V[-2, 1:-1]) * dx
)

net_boundary_flux = (
    flux_left
    + flux_right
    + flux_bottom
    + flux_top
)

boundary_audit_divergence = backward_divergence(U, V)

integrated_divergence = (
    np.sum(
        boundary_audit_divergence[fluid_center]
    )
    * dx
    * dy
)

flux_divergence_mismatch = (
    net_boundary_flux - integrated_divergence
)

# Normalize by the total inward boundary flux.
total_inward_flux = sum(
    max(-boundary_flux, 0.0)
    for boundary_flux in (
        flux_left,
        flux_right,
        flux_bottom,
        flux_top
    )
)

relative_boundary_imbalance = (
    net_boundary_flux
    / max(total_inward_flux, 1.0e-14)
)

print("\nFull-domain boundary audit:")
print(f"Left outward flux:   {flux_left:.8e}")
print(f"Right outward flux:  {flux_right:.8e}")
print(f"Bottom outward flux: {flux_bottom:.8e}")
print(f"Top outward flux:    {flux_top:.8e}")
print(f"Net outward flux:    {net_boundary_flux:.8e}")

print(
    f"Relative full-boundary imbalance: "
    f"{relative_boundary_imbalance:.8e}"
)

print(
    f"Integrated fluid divergence: "
    f"{integrated_divergence:.8e}"
)

print(
    f"Flux-divergence mismatch: "
    f"{flux_divergence_mismatch:.8e}"
)

# Final Divergence

final_divergence = backward_divergence(
    U,
    V
)


valid_divergence_cells = (
    fluid_center
    &
    fluid_e
    &
    fluid_w
    &
    fluid_n
    &
    fluid_s
).copy()


valid_divergence_cells[0, :] = False
valid_divergence_cells[-1, :] = False

valid_divergence_cells[:, 0] = False
valid_divergence_cells[:, -1] = False


valid_divergence = (
    final_divergence[
        valid_divergence_cells
    ]
)


all_fluid_divergence = (
    final_divergence[
        fluid_center
    ]
)


absolute_divergence = np.abs(
    final_divergence
).copy()

absolute_divergence[
    ~fluid_center
] = 0.0


max_index = np.unravel_index(
    np.argmax(
        absolute_divergence
    ),
    absolute_divergence.shape
)


max_y_index = (
    max_index[0] + 1
)

max_x_index = (
    max_index[1] + 1
)


# Final Mass Flow

(
    Q_in,
    Q_out,
    mass_error,
    relative_mass_error
) = calculate_fluxes(
    U
)


# Basic Results

print()

print(
    f"Grid: {nx} x {ny}"
)

print(
    f"dx = {dx:.8f}"
)

print(
    f"dy = {dy:.8f}"
)

print(
    f"Total simulation time = "
    f"{completed_steps * dt:.6f}"
)

print(
    f"Completed steps = "
    f"{completed_steps}"
)

print(
    f"Steady-state stop = "
    f"{steady_state_reached}"
)

print()

print(
    "NACA 0012 CFD ANALYSIS"
)

print()

print(
    "Airfoil scaling:"
)

print(
    f"Base chord = "
    f"{base_chord:.6f}"
)

print(
    f"Scale factor = "
    f"{scale_factor:.6f}"
)

print(
    f"Final chord = "
    f"{chord:.6f}"
)

print()

print(
    "Geometry:"
)

print(
    f"Leading edge = "
    f"{x_le:.6f}"
)

print(
    f"Trailing edge = "
    f"{x_le + chord:.6f}"
)

print(
    f"Airfoil cells per chord = "
    f"{chord / dx:.2f}"
)

print(
    f"Angle of attack = "
    f"{angle_of_attack:.2f} degrees"
)

print(
    f"Thickness ratio = "
    f"{t:.4f}"
)

print()

print(
    "Freestream:"
)

print(
    f"Inlet speed = "
    f"{U_inf_input:.6f}"
)

print(
    f"U_inlet = "
    f"{U_inlet:.6f}"
)

print(
    f"V_inlet = "
    f"{V_inlet:.6f}"
)

print(
    f"U_inf = "
    f"{U_inf:.6f}"
)

q_inf = (
    0.5
    *
    rho
    *
    U_inf**2
)

print(
    f"Dynamic pressure = "
    f"{q_inf:.6f}"
)

Re = (
    U_inf
    *
    chord
    /
    nu
)

print(
    f"Reynolds number = "
    f"{Re:.6f}"
)

print()

print(
    "Velocity:"
)

speed = np.sqrt(
    U**2
    +
    V**2
)

print(
    f"Maximum speed = "
    f"{np.max(speed):.8f}"
)

print(
    f"Minimum speed = "
    f"{np.min(speed):.8f}"
)

print()

print(
    "Global mass conservation:"
)

print(
    f"Q_in = "
    f"{Q_in:.12f}"
)

print(
    f"Q_out = "
    f"{Q_out:.12f}"
)

print(
    f"Q_out - Q_in = "
    f"{mass_error:.12e}"
)

print(
    f"Relative mass error = "
    f"{relative_mass_error:.12e}"
)

print()

print(
    "Divergence:"
)

print(
    f"Maximum divergence, "
    f"all fluid cells = "
    f"{np.max(np.abs(all_fluid_divergence)):.8e}"
)

print(
    f"Maximum divergence, "
    f"valid cells = "
    f"{np.max(np.abs(valid_divergence)):.8e}"
)

print(
    f"RMS divergence, "
    f"valid cells = "
    f"{np.sqrt(np.mean(valid_divergence**2)):.8e}"
)

print()

print(
    "Maximum-divergence location:"
)

print(
    f"x = "
    f"{X[max_y_index, max_x_index]:.8f}"
)

print(
    f"y = "
    f"{Y[max_y_index, max_x_index]:.8f}"
)

print(
    f"divergence = "
    f"{final_divergence[max_index]:.8e}"
)

print()

print(
    "Pressure solver:"
)

print(
    f"Last SOR iterations = "
    f"{last_pressure_iterations}"
)

print(
    f"Last Poisson residual = "
    f"{last_pressure_residual:.8e}"
)


# Stagger-aware symmetry diagnostics

pressure_pair_mask = fluid & np.flipud(fluid)
u_pair_mask = u_face_open & np.flipud(u_face_open)
v_pair_mask = v_face_open & np.flipud(v_face_open)

pressure_symmetry_error = np.max(
    np.abs(P - np.flipud(P))[pressure_pair_mask]
)

u_symmetry_error = np.max(
    np.abs(
        U[:, :-1] - np.flipud(U[:, :-1])
    )[u_pair_mask]
)

v_antisymmetry_error = np.max(
    np.abs(
        V[:-1, :] + np.flipud(V[:-1, :])
    )[v_pair_mask]
)

print()
print("Stagger-aware symmetry diagnostics:")
print(
    f"Maximum U symmetry error = "
    f"{u_symmetry_error:.8e}"
)
print(
    f"Maximum V antisymmetry error = "
    f"{v_antisymmetry_error:.8e}"
)
print(
    f"Maximum P symmetry error = "
    f"{pressure_symmetry_error:.8e}"
)


# Control Volume Force Analysis

mu = rho * nu


# Convert the staggered face velocities to the pressure-cell
# centers used by the control-volume integration.

def velocities_at_cell_centers(U_face, V_face):

    U_cell = U_face.copy()
    V_cell = V_face.copy()

    U_cell[:, 1:-1] = 0.5 * (
        U_face[:, 1:-1]
        +
        U_face[:, :-2]
    )

    V_cell[1:-1, :] = 0.5 * (
        V_face[1:-1, :]
        +
        V_face[:-2, :]
    )

    V_cell[-1, :] = V_face[-2, :]

    return U_cell, V_cell


U_force, V_force = velocities_at_cell_centers(
    U,
    V
)

(
    U_before_final_force,
    V_before_final_force
) = velocities_at_cell_centers(
    U_before_final_step,
    V_before_final_step
)

control_volumes = [
    {
        "name": "Inner",
        "x_left": x_le - 0.25 * chord,
        "x_right": x_le + 1.25 * chord,
        "y_bottom": y_center - 0.625 * chord,
        "y_top": y_center + 0.625 * chord
    },

    {
        "name": "Middle",
        "x_left": x_le - 0.375 * chord,
        "x_right": x_le + 1.375 * chord,
        "y_bottom": y_center - 0.75 * chord,
        "y_top": y_center + 0.75 * chord
    },

    {
        "name": "Outer",
        "x_left": x_le - 0.50 * chord,
        "x_right": x_le + 1.50 * chord,
        "y_bottom": y_center - 0.875 * chord,
        "y_top": y_center + 0.875 * chord
    }
]

airfoil_x_min = x_le
airfoil_x_max = x_le + chord
airfoil_y_min = np.min(y_lower)
airfoil_y_max = np.max(y_upper)

if airfoil_x_min <= domain_x_min or airfoil_x_max >= domain_x_max:
    raise ValueError(
        "The scaled airfoil does not fit inside the x-domain. "
        "Increase the domain size or move x_le."
    )

if airfoil_y_min <= domain_y_min or airfoil_y_max >= domain_y_max:
    raise ValueError(
        "The scaled airfoil does not fit inside the y-domain. "
        "Increase the domain size or move y_center."
    )

for cv in control_volumes:
    if (
        cv["x_left"] <= domain_x_min
        or cv["x_right"] >= domain_x_max
        or cv["y_bottom"] <= domain_y_min
        or cv["y_top"] >= domain_y_max
    ):
        raise ValueError(
            f'{cv["name"]} control volume does not fit inside the domain. '
            "Increase the domain size, reduce scale_factor, or reposition the airfoil."
        )

dU_dx_cv = np.zeros_like(U_force)
dU_dy_cv = np.zeros_like(U_force)
dV_dx_cv = np.zeros_like(V_force)
dV_dy_cv = np.zeros_like(V_force)

dU_dx_cv[:, 1:-1] = (
    U_force[:, 2:]
    - U_force[:, :-2]
) / (2.0 * dx)

dU_dy_cv[1:-1, :] = (
    U_force[2:, :]
    - U_force[:-2, :]
) / (2.0 * dy)

dV_dx_cv[:, 1:-1] = (
    V_force[:, 2:]
    - V_force[:, :-2]
) / (2.0 * dx)

dV_dy_cv[1:-1, :] = (
    V_force[2:, :]
    - V_force[:-2, :]
) / (2.0 * dy)

tau_xx = 2.0 * mu * dU_dx_cv
tau_yy = 2.0 * mu * dV_dy_cv
tau_xy = mu * (dU_dy_cv + dV_dx_cv)

dU_dt_final = (
    U_force - U_before_final_force
) / dt

dV_dt_final = (
    V_force - V_before_final_force
) / dt

def calculate_control_volume_force(
    cv_x_left_target,
    cv_x_right_target,
    cv_y_bottom_target,
    cv_y_top_target
):

    cv_j_left = np.argmin(
        np.abs(x - cv_x_left_target)
    )

    cv_j_right = np.argmin(
        np.abs(x - cv_x_right_target)
    )

    cv_i_bottom = np.argmin(
        np.abs(y - cv_y_bottom_target)
    )

    cv_i_top = np.argmin(
        np.abs(y - cv_y_top_target)
    )

    if cv_j_left >= cv_j_right:
        raise ValueError(
            "Control-volume x limits are invalid."
        )

    if cv_i_bottom >= cv_i_top:
        raise ValueError(
            "Control-volume y limits are invalid."
        )

    cv_x_left = x[cv_j_left]
    cv_x_right = x[cv_j_right]
    cv_y_bottom = y[cv_i_bottom]
    cv_y_top = y[cv_i_top]

    solid_on_left = np.any(
        solid[
            cv_i_bottom:cv_i_top + 1,
            cv_j_left
        ]
    )

    solid_on_right = np.any(
        solid[
            cv_i_bottom:cv_i_top + 1,
            cv_j_right
        ]
    )

    solid_on_bottom = np.any(
        solid[
            cv_i_bottom,
            cv_j_left:cv_j_right + 1
        ]
    )

    solid_on_top = np.any(
        solid[
            cv_i_top,
            cv_j_left:cv_j_right + 1
        ]
    )

    if (
        solid_on_left
        or solid_on_right
        or solid_on_bottom
        or solid_on_top
    ):
        raise ValueError(
            "Control volume intersects the solid airfoil."
        )

    y_cv = y[cv_i_bottom:cv_i_top + 1]
    x_cv = x[cv_j_left:cv_j_right + 1]

    U_left = U_force[
        cv_i_bottom:cv_i_top + 1,
        cv_j_left
    ]
    V_left = V_force[
        cv_i_bottom:cv_i_top + 1,
        cv_j_left
    ]
    P_left = P[
        cv_i_bottom:cv_i_top + 1,
        cv_j_left
    ]

    U_right = U_force[
        cv_i_bottom:cv_i_top + 1,
        cv_j_right
    ]
    V_right = V_force[
        cv_i_bottom:cv_i_top + 1,
        cv_j_right
    ]
    P_right = P[
        cv_i_bottom:cv_i_top + 1,
        cv_j_right
    ]

    U_bottom = U_force[
        cv_i_bottom,
        cv_j_left:cv_j_right + 1
    ]
    V_bottom = V_force[
        cv_i_bottom,
        cv_j_left:cv_j_right + 1
    ]
    P_bottom = P[
        cv_i_bottom,
        cv_j_left:cv_j_right + 1
    ]

    U_top = U_force[
        cv_i_top,
        cv_j_left:cv_j_right + 1
    ]
    V_top = V_force[
        cv_i_top,
        cv_j_left:cv_j_right + 1
    ]
    P_top = P[
        cv_i_top,
        cv_j_left:cv_j_right + 1
    ]

    Fx_pressure_cv = (
        np.trapezoid(P_left, y_cv)
        - np.trapezoid(P_right, y_cv)
    )

    Fy_pressure_cv = (
        np.trapezoid(P_bottom, x_cv)
        - np.trapezoid(P_top, x_cv)
    )

    Fx_viscous_cv = (
        np.trapezoid(
            -tau_xx[
                cv_i_bottom:cv_i_top + 1,
                cv_j_left
            ],
            y_cv
        )
        + np.trapezoid(
            tau_xx[
                cv_i_bottom:cv_i_top + 1,
                cv_j_right
            ],
            y_cv
        )
        + np.trapezoid(
            -tau_xy[
                cv_i_bottom,
                cv_j_left:cv_j_right + 1
            ],
            x_cv
        )
        + np.trapezoid(
            tau_xy[
                cv_i_top,
                cv_j_left:cv_j_right + 1
            ],
            x_cv
        )
    )

    Fy_viscous_cv = (
        np.trapezoid(
            -tau_xy[
                cv_i_bottom:cv_i_top + 1,
                cv_j_left
            ],
            y_cv
        )
        + np.trapezoid(
            tau_xy[
                cv_i_bottom:cv_i_top + 1,
                cv_j_right
            ],
            y_cv
        )
        + np.trapezoid(
            -tau_yy[
                cv_i_bottom,
                cv_j_left:cv_j_right + 1
            ],
            x_cv
        )
        + np.trapezoid(
            tau_yy[
                cv_i_top,
                cv_j_left:cv_j_right + 1
            ],
            x_cv
        )
    )

    Fx_momentum_cv = (
        np.trapezoid(
            rho * U_left**2,
            y_cv
        )
        - np.trapezoid(
            rho * U_right**2,
            y_cv
        )
        + np.trapezoid(
            rho * U_bottom * V_bottom,
            x_cv
        )
        - np.trapezoid(
            rho * U_top * V_top,
            x_cv
        )
    )

    Fy_momentum_cv = (
        np.trapezoid(
            rho * U_left * V_left,
            y_cv
        )
        - np.trapezoid(
            rho * U_right * V_right,
            y_cv
        )
        + np.trapezoid(
            rho * V_bottom**2,
            x_cv
        )
        - np.trapezoid(
            rho * V_top**2,
            x_cv
        )
    )

    dU_dt_inside_cv = dU_dt_final[
        cv_i_bottom:cv_i_top + 1,
        cv_j_left:cv_j_right + 1
    ]

    dV_dt_inside_cv = dV_dt_final[
        cv_i_bottom:cv_i_top + 1,
        cv_j_left:cv_j_right + 1
    ]

    integrated_dU_dt_x = np.trapezoid(
        dU_dt_inside_cv,
        x_cv,
        axis=1
    )

    integrated_dV_dt_x = np.trapezoid(
        dV_dt_inside_cv,
        x_cv,
        axis=1
    )

    Fx_unsteady_cv = (
        -rho
        * np.trapezoid(
            integrated_dU_dt_x,
            y_cv
        )
    )

    Fy_unsteady_cv = (
        -rho
        * np.trapezoid(
            integrated_dV_dt_x,
            y_cv
        )
    )

    Fx_total_cv = (
        Fx_pressure_cv
        + Fx_viscous_cv
        + Fx_momentum_cv
        + Fx_unsteady_cv
    )

    Fy_total_cv = (
        Fy_pressure_cv
        + Fy_viscous_cv
        + Fy_momentum_cv
        + Fy_unsteady_cv
    )

    drag_cv = (
        Fx_total_cv * np.cos(alpha)
        + Fy_total_cv * np.sin(alpha)
    )

    lift_cv = (
        -Fx_total_cv * np.sin(alpha)
        + Fy_total_cv * np.cos(alpha)
    )

    Cd_cv = drag_cv / (q_inf * chord)
    Cl_cv = lift_cv / (q_inf * chord)

    return {
        "x_left": cv_x_left,
        "x_right": cv_x_right,
        "y_bottom": cv_y_bottom,
        "y_top": cv_y_top,
        "Fx_pressure": Fx_pressure_cv,
        "Fy_pressure": Fy_pressure_cv,
        "Fx_viscous": Fx_viscous_cv,
        "Fy_viscous": Fy_viscous_cv,
        "Fx_momentum": Fx_momentum_cv,
        "Fy_momentum": Fy_momentum_cv,
        "Fx_unsteady": Fx_unsteady_cv,
        "Fy_unsteady": Fy_unsteady_cv,
        "Fx_total": Fx_total_cv,
        "Fy_total": Fy_total_cv,
        "drag": drag_cv,
        "lift": lift_cv,
        "Cd": Cd_cv,
        "Cl": Cl_cv
    }

cv_results = []

for cv in control_volumes:
    result = calculate_control_volume_force(
        cv["x_left"],
        cv["x_right"],
        cv["y_bottom"],
        cv["y_top"]
    )

    result["name"] = cv["name"]
    cv_results.append(result)

Cd_values = np.array([
    result["Cd"]
    for result in cv_results
])

Cl_values = np.array([
    result["Cl"]
    for result in cv_results
])

Cd_mean = np.mean(Cd_values)
Cl_mean = np.mean(Cl_values)

Cd_spread = (
    np.max(Cd_values)
    - np.min(Cd_values)
)

Cl_spread = (
    np.max(Cl_values)
    - np.min(Cl_values)
)

Cd_relative_spread = (
    Cd_spread
    / max(abs(Cd_mean), 1.0e-14)
)

reference_result = next(
    result
    for result in cv_results
    if result["name"] == "Middle"
)

Fx_total = reference_result["Fx_total"]
Fy_total = reference_result["Fy_total"]
drag = reference_result["drag"]
lift = reference_result["lift"]
Cd = reference_result["Cd"]
Cl = reference_result["Cl"]

print()
print("Control-volume force comparison:")

for result in cv_results:
    print()
    print(
        f'{result["name"]} control volume:'
    )
    print(
        f'x = {result["x_left"]:.4f} to '
        f'{result["x_right"]:.4f}'
    )
    print(
        f'y = {result["y_bottom"]:.4f} to '
        f'{result["y_top"]:.4f}'
    )
    print(
        f'Fx_pressure = '
        f'{result["Fx_pressure"]:.8f}'
    )
    print(
        f'Fx_viscous = '
        f'{result["Fx_viscous"]:.8f}'
    )
    print(
        f'Fx_momentum = '
        f'{result["Fx_momentum"]:.8f}'
    )
    print(
        f'Fx_unsteady = '
        f'{result["Fx_unsteady"]:.8f}'
    )
    print(
        f'Fy_total = '
        f'{result["Fy_total"]:.8f}'
    )
    print(
        f'Drag = {result["drag"]:.8f}'
    )
    print(
        f'Lift = {result["lift"]:.8f}'
    )
    print(
        f'Cd = {result["Cd"]:.8f}'
    )
    print(
        f'Cl = {result["Cl"]:.8f}'
    )

print()
print("Control-volume consistency:")
print(
    f"Mean Cd = {Cd_mean:.8f}"
)
print(
    f"Cd spread = {Cd_spread:.8f}"
)
print(
    f"Relative Cd spread = "
    f"{100.0 * Cd_relative_spread:.4f}%"
)
print(
    f"Mean Cl = {Cl_mean:.8f}"
)
print(
    f"Cl spread = {Cl_spread:.8f}"
)

print()
print("Reference aerodynamic result:")
print("Using the Middle control volume")
print(
    f"Drag = {drag:.8f}"
)
print(
    f"Lift = {lift:.8f}"
)
print(
    f"Cd = {Cd:.8f}"
)
print(
    f"Cl = {Cl:.8f}"
)

# Wake Analysis

wake_x_target = min(
    x_le + 1.50 * chord,
    domain_x_max - dx
)

wake_x_location = np.argmin(
    np.abs(
        x - wake_x_target
    )
)

wake_velocity = U[
    :,
    wake_x_location
]

wake_deficit = (
    U_inf
    -
    wake_velocity
)

print()

print(
    "Wake analysis:"
)

print(
    f"Wake x location = "
    f"{x[wake_x_location]:.6f}"
)

print(
    f"Freestream velocity = "
    f"{U_inf:.6f}"
)

print(
    f"Minimum wake velocity = "
    f"{np.min(wake_velocity):.6f}"
)

print(
    f"Maximum velocity deficit = "
    f"{np.max(wake_deficit):.6f}"
)


# Pressure Distribution

num_pressure_points = 100

x_pressure = np.linspace(
    x_surface[0],
    x_surface[-1],
    num_pressure_points
)

pressure_upper = np.zeros(
    num_pressure_points
)

pressure_lower = np.zeros(
    num_pressure_points
)

sample_distance = max(
    dx,
    dy
)


# We retain the same nearest-fluid-cell sampling method,
# but construct the distance fields efficiently.

for k, x_point in enumerate(x_pressure):

    surface_index = np.argmin(
        np.abs(
            x_surface
            -
            x_point
        )
    )

    y_upper_point = (
        y_upper[
            surface_index
        ]
    )

    y_lower_point = (
        y_lower[
            surface_index
        ]
    )


    upper_distance_squared = (
        (X - x_point)**2
        +
        (
            Y
            -
            (
                y_upper_point
                +
                sample_distance
            )
        )**2
    )

    upper_distance_squared[
        solid
    ] = np.inf

    upper_index = np.unravel_index(
        np.argmin(
            upper_distance_squared
        ),
        upper_distance_squared.shape
    )

    pressure_upper[k] = P[
        upper_index
    ]


    lower_distance_squared = (
        (X - x_point)**2
        +
        (
            Y
            -
            (
                y_lower_point
                -
                sample_distance
            )
        )**2
    )

    lower_distance_squared[
        solid
    ] = np.inf

    lower_index = np.unravel_index(
        np.argmin(
            lower_distance_squared
        ),
        lower_distance_squared.shape
    )

    pressure_lower[k] = P[
        lower_index
    ]


freestream_region = (
    (X >= domain_x_min + 0.10 * (domain_x_max - domain_x_min))
    &
    (X <= x_le - 0.75 * chord)
    &
    fluid
)

if not np.any(freestream_region):
    raise ValueError(
        "Freestream pressure reference region is empty. "
        "Increase the upstream domain distance or move the airfoil downstream."
    )

P_inf = np.mean(
    P[freestream_region]
)

Cp_upper = (
    pressure_upper
    -
    P_inf
) / q_inf

Cp_lower = (
    pressure_lower
    -
    P_inf
) / q_inf


x_over_c = (
    x_pressure
    -
    x_le
) / chord


pressure_difference = (
    pressure_lower
    -
    pressure_upper
)

Cp_difference = (
    Cp_lower
    -
    Cp_upper
)


print()

print(
    "Pressure distribution:"
)

print(
    f"Freestream reference pressure = "
    f"{P_inf:.8f}"
)

print(
    f"Maximum Cp upper = "
    f"{np.max(Cp_upper):.6f}"
)

print(
    f"Minimum Cp upper = "
    f"{np.min(Cp_upper):.6f}"
)

print(
    f"Maximum Cp lower = "
    f"{np.max(Cp_lower):.6f}"
)

print(
    f"Minimum Cp lower = "
    f"{np.min(Cp_lower):.6f}"
)

print(
    f"Maximum Cp difference = "
    f"{np.max(Cp_difference):.6f}"
)

print(
    f"Minimum Cp difference = "
    f"{np.min(Cp_difference):.6f}"
)

print()

print(
    "Final CFD summary:"
)

print(
    f"Scale factor: "
    f"{scale_factor:.4f}"
)

print(
    f"Chord: "
    f"{chord:.4f}"
)

print(
    f"Angle of attack: "
    f"{angle_of_attack:.2f} degrees"
)

print(
    f"Reynolds number: "
    f"{Re:.6f}"
)

print(
    f"Lift: "
    f"{lift:.6f}"
)

print(
    f"Drag: "
    f"{drag:.6f}"
)

print(
    f"Cl: "
    f"{Cl:.6f}"
)

print(
    f"Cd: "
    f"{Cd:.6f}"
)

output_folder = Path(__file__).resolve().parent / "pressure_audit_results"
output_folder.mkdir(exist_ok=True)

U_cell, V_cell = velocities_at_cell_centers(U, V)

output_file = output_folder / (
    f"flow_{nx}x{ny}_"
    f"aoa_{angle_of_attack:+.1f}_"
    f"T_{completed_steps * dt:.2f}.npz"
)

np.savez_compressed(
    output_file,
    x=x,
    y=y,
    U_face=U,
    V_face=V,
    U_cell=U_cell,
    V_cell=V_cell,
    P=P,
    solid=solid,
    x_surface=x_surface,
    y_upper=y_upper,
    y_lower=y_lower,
    x_over_c=x_over_c,
    Cp_upper=Cp_upper,
    Cp_lower=Cp_lower,
    P_inf=P_inf,
    q_inf=q_inf,
    U_inf=U_inf,
    alpha_degrees=angle_of_attack,
    chord=chord,
    x_le=x_le,
    y_center=y_center,
    rho=rho,
    nu=nu,
    dt=dt,
    completed_steps=completed_steps,
    simulation_time=completed_steps * dt,
    pressure_max_iterations=pressure_max_iterations,
    pressure_residual_tolerance=pressure_residual_tolerance,
    pressure_failed_steps=pressure_failed_steps,
    sparse_pressure_solves=pressure_fallback.calls,
    pressure_backend="SOR then cached sparse LU on tolerance failure",
    pressure_total_iterations=pressure_total_iterations,
    pressure_worst_residual=pressure_worst_residual,
    last_pressure_residual=last_pressure_residual,
    middle_Fx_unsteady=reference_result["Fx_unsteady"],
    Cd_relative_spread=Cd_relative_spread,
    force_history=np.asarray(force_history),
    force_history_columns=np.array(["T", "Cd", "Cl", "Fx_unsteady", "Fy_unsteady", "Cd_relative_spread"]),
    u_wall_count=u_wall_count,
    v_wall_count=v_wall_count,
    Cd=Cd,
    Cl=Cl,
)

print(f"\nSaved data: {output_file}")
# Plots' to its end.
# This section uses the simulation in memory; it does not load NPZ files.
# Keep SHOW_PLOTS = True in your solver settings.

# Plots
if SHOW_PLOTS:
    from pathlib import Path
    from matplotlib.colors import TwoSlopeNorm, ListedColormap
    from matplotlib.patches import Patch

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'figure.titlesize': 16, 'axes.spines.top': False,
        'axes.spines.right': False, 'legend.frameon': False,
        'lines.linewidth': 2, 'savefig.dpi': 220,
    })
    # Preserve the original staggered U and V arrays.
    plot_u, plot_v = velocities_at_cell_centers(U, V)
    plot_u = np.ma.array(plot_u, mask=solid)
    plot_v = np.ma.array(plot_v, mask=solid)
    plot_speed = np.hypot(plot_u, plot_v) / U_inf
    plot_cp = np.ma.array((P - P_inf) / q_inf, mask=solid)
    xc = (x - x_le) / chord
    yc = (y - y_center) / chord
    sx = (x_surface - x_le) / chord
    sy_upper = (y_upper - y_center) / chord
    sy_lower = (y_lower - y_center) / chord
    run_label = (f'NACA 0012 | angle {angle_of_attack:g}° | Re {Re:g} | '
                 f'{nx} × {ny} grid | T = {completed_steps * dt:g}')
    chart_folder = Path(__file__).resolve().parent / 'pressure_audit_figures' / (
        f'{nx}x{ny}_angle_{angle_of_attack:+.1f}_T_{completed_steps*dt:.2f}')
    chart_folder.mkdir(parents=True, exist_ok=True)

    def finish_chart(fig, name, note):
        fig.text(.07, .025, note, fontsize=9, color='#454545')
        fig.tight_layout(rect=(0, .075, 1, .93), pad=1.6, h_pad=4.0, w_pad=2.5)
        for extension in ('png', 'pdf'):
            fig.savefig(chart_folder / f'{name}.{extension}')

    def flow_axes(ax):
        ax.contourf(xc, yc, solid.astype(float), levels=[.5, 1.5],
                    colors=['#313b45'], zorder=5)
        ax.plot(sx, sy_upper, color='black', lw=.9, zorder=6)
        ax.plot(sx, sy_lower, color='black', lw=.9, zorder=6)
        ax.set(xlim=(max(xc[0], -.75), min(xc[-1], 2.25)),
               ylim=(max(yc[0], -.9), min(yc[-1], .9)),
               xlabel='Chord position, (x − leading edge) / chord',
               ylabel='Height from airfoil center / chord')
        ax.set_aspect('equal')

    # FIGURE 1: all four original flow-field views.
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Flow around the airfoil\n' + run_label)
    speed_max = max(1.3, np.ceil(float(plot_speed.max()) * 10) / 10)
    speed_levels = np.linspace(0, speed_max, 40)
    umin = min(0, np.floor(float((plot_u/U_inf).min()) * 10) / 10)
    umax = max(1.3, np.ceil(float((plot_u/U_inf).max()) * 10) / 10)
    im = axs[0, 0].contourf(xc, yc, plot_u/U_inf,
                           levels=np.linspace(umin, umax, 40), cmap='viridis')
    axs[0, 0].set_title('Horizontal velocity')
    fig.colorbar(im, ax=axs[0, 0], shrink=.8, label='Horizontal velocity / inlet speed')
    cp_limit = max(1.2, np.ceil(float(np.max(np.abs(plot_cp))) * 5) / 5)
    im = axs[0, 1].contourf(xc, yc, plot_cp,
                           levels=np.linspace(-cp_limit, cp_limit, 49),
                           cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-cp_limit, vcenter=0, vmax=cp_limit))
    axs[0, 1].set_title('Pressure relative to upstream reference')
    fig.colorbar(im, ax=axs[0, 1], shrink=.8, label='Pressure coefficient, Cp')
    for ax in axs[1]:
        im = ax.contourf(xc, yc, plot_speed, levels=speed_levels, cmap='viridis')
        fig.colorbar(im, ax=ax, shrink=.8, label='Speed / inlet speed')
    stride = max(1, min(nx, ny)//25)
    axs[1, 0].quiver(xc[::stride], yc[::stride],
                     plot_u[::stride, ::stride]/U_inf, plot_v[::stride, ::stride]/U_inf,
                     color='white', scale=22, width=.004)
    axs[1, 0].set_title('Speed and velocity arrows')
    axs[1, 1].streamplot(xc, yc, plot_u, plot_v, density=1.2,
                         color='white', linewidth=.7, arrowsize=.8)
    axs[1, 1].set_title('Speed and streamlines')
    for ax in axs.flat:
        flow_axes(ax)
    finish_chart(fig, '01_flow_fields',
                 'Dark region: numerical solid mask. Black outline: intended airfoil. '
                 'Speed ratio 1 = inlet speed.\n'
                 'Pressure: red is above the reference, blue is below it. '
                 'Coordinates are normalized by chord.')

    # FIGURE 2: original Cp and pressure-difference charts.
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Pressure above and below the airfoil\n' + run_label)
    axs[0].plot(x_over_c, Cp_upper, color='#1764ab', label='Upper side')
    axs[0].plot(x_over_c, Cp_lower, '--', color='#bb4b16', label='Lower side')
    axs[0].invert_yaxis()
    axs[0].set(title='Near-surface Cp · negative values plotted upward', ylabel='Pressure coefficient, Cp')
    axs[0].legend()
    axs[1].plot(x_over_c, Cp_lower-Cp_upper, color='#1764ab')
    axs[1].set(title='Pressure difference between sides', ylabel='Cp lower − Cp upper')
    for ax in axs:
        ax.axhline(0, color='gray', lw=.8, ls=':')
        ax.set(xlim=(0, 1), xlabel='Chord position, x/c (0 = front, 1 = trailing edge)')
        ax.grid(alpha=.2)
    finish_chart(fig, '02_pressure_loading',
                 'Samples use the nearest fluid cell to targets one grid spacing above/below the geometry; '
                 'they are not wall-pressure measurements.\n'
                 'Positive difference means higher pressure below. Lines connect samples without smoothing.')

    # FIGURE 3: wake, profile development, and numerical geometry.
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Wake and airfoil resolution\n' + run_label)
    wake_j = int(np.argmin(abs(x-(x_le+1.5*chord))))
    wake_parallel = (plot_u[:, wake_j]*np.cos(alpha) + plot_v[:, wake_j]*np.sin(alpha))/U_inf
    axs[0].plot(wake_parallel, yc, label='Final wake')
    axs[0].axvline(1, color='gray', ls='--', label='Inlet speed')
    axs[0].set(title=f'Final wake at x = {x[wake_j]:.4f}',
               xlabel='Velocity along inlet direction / inlet speed', ylabel='Height from airfoil center / chord')
    axs[0].legend()
    keys = sorted(profiles)
    chosen = [keys[i] for i in np.unique(np.linspace(0, len(keys)-1, min(5, len(keys))).astype(int))] if keys else []
    for k in chosen:
        uf = profiles[k]
        line = .5*(uf[:, wake_j] + uf[:, wake_j-1]) / U_inf
        axs[1].plot(line, yc, label=f'T = {k*dt:g}')
    axs[1].axvline(U_inlet/U_inf, color='gray', ls=':')
    axs[1].set(title='Wake development at the same location',
               xlabel='Horizontal velocity / inlet speed', ylabel='Height from airfoil center / chord')
    if chosen: axs[1].legend(title='Selected saved times', fontsize=9)
    for ax in axs[:2]:
        ax.set_ylim(-1.5, 1.5); ax.grid(alpha=.2)
    axs[2].pcolormesh(xc, yc, solid.astype(int), shading='nearest',
                      cmap=ListedColormap(['#eef3f7', '#313b45']), vmin=0, vmax=1)
    axs[2].plot(sx, sy_upper, color='#bb4b16', label='Intended airfoil')
    axs[2].plot(sx, sy_lower, color='#bb4b16')
    axs[2].set(xlim=(-.1, 1.1), ylim=(-.22, .22),
               xlabel='Chord position, x/c', ylabel='Height / chord',
               title=f'Airfoil mask · {chord/dx:.1f} cells per chord')
    axs[2].set_aspect('equal')
    axs[2].legend(handles=[Patch(color='#313b45', label='Solid grid cells'),
                          axs[2].lines[0]], loc='upper center', fontsize=9)
    finish_chart(fig, '03_wake_and_geometry',
                 'Final wake uses velocity along the inlet direction. Development uses horizontal velocity '
                 '(only U was stored in the history).\n'
                 'At nonzero angles these are different quantities. The mask view shows how the grid represents the airfoil.')

    # FIGURE 4: all original histories, plus pressure-solver effort.
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('How the calculation settles over time\n' + run_label)
    axs[0, 0].plot(time_history, np.asarray(max_velocity_history)/U_inf, color='#1764ab')
    axs[0, 0].set(title='Maximum cell-centered speed', ylabel='Maximum speed / inlet speed')
    axs[0, 1].semilogy(time_history, np.maximum(np.abs(divergence_history), 1e-16), color='#1764ab')
    axs[0, 1].axhline(target_divergence, color='#bb4b16', ls='--', label='Solver target')
    axs[0, 1].set(title='Divergence · smaller is better', ylabel='Maximum absolute divergence')
    axs[0, 1].legend()
    axs[1, 0].semilogy(time_history, np.maximum(np.abs(relative_mass_error_history), 1e-16), color='#1764ab')
    axs[1, 0].set(title='Inlet–outlet flow imbalance', ylabel='Absolute relative flow imbalance')
    axs[1, 1].plot(time_history, pressure_iterations_history, color='#1764ab', drawstyle='steps-mid')
    axs[1, 1].axhline(pressure_max_iterations, color='#bb4b16', ls='--', label='Iteration cap')
    axs[1, 1].set(title='SOR effort (sparse solves reported in terminal)', ylabel='Iterations per sampled timestep')
    axs[1, 1].legend()
    for ax in axs.flat:
        ax.set_xlabel('Simulated time, T');ax.set_xlim(0, completed_steps*dt);ax.grid(alpha=.2)
    finish_chart(fig, '04_convergence_history',
                 'Lines connect recorded samples. The steady stopping check still uses the original speed diagnostic.\n'
                 'Low final errors support numerical consistency; these charts alone do not establish physical accuracy.')
    print(f'\nCharts saved to: {chart_folder}')
    # Display all figures together after saving the force history.

# Force history is also available as a plain CSV.
force_array = np.asarray(force_history)
np.savetxt(output_file.with_suffix(".forces.csv"), force_array, delimiter=",",
           header="T,Cd,Cl,Fx_unsteady,Fy_unsteady,Cd_relative_spread", comments="")
if len(force_array) >= 2:
    tail = force_array[force_array[:, 0] >= force_array[-1, 0] - 1.0]
    if len(tail) >= 2:
        print(f"Force variation over recorded T={tail[0,0]:.3f} to {tail[-1,0]:.3f}:")
        print(f"Cd peak-to-peak / mean: {np.ptp(tail[:,1]) / max(abs(np.mean(tail[:,1])), 1e-14):.6e}")
        print(f"Cl absolute peak-to-peak: {np.ptp(tail[:,2]):.6e}")
        print("These are diagnostics, not an automatic steady-state verdict.")
if SHOW_PLOTS:
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col, label in zip(axs.flat, [1,2,3,4],
                             ["Drag coefficient, Cd", "Lift coefficient, Cl", "Unsteady x-force", "Unsteady y-force"]):
        ax.plot(force_array[:,0], force_array[:,col])
        ax.set(xlabel="Simulated time, T", ylabel=label)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Middle control-volume forces | angle {angle_of_attack:g}° | {nx} × {ny}")
    fig.tight_layout(rect=(0,0,1,0.94))
    fig.savefig(output_folder / (output_file.stem + "_forces.png"), dpi=180)
    plt.show()
