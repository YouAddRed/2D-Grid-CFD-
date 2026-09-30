"""Shared predictor and pressure routines. Boundary conditions are supplied by each case."""
import numpy as np

class CFDCore:
    def __init__(self, *, dx, dy, dt, dt_nu, inv_dx2, inv_dy2, u_wall_count, v_wall_count, a_e, a_w, a_n, a_s, pressure_denominator, fluid_center, red_cells, black_cells, pressure_max_iterations, pressure_residual_tolerance, sor_omega, apply_pressure_bc):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.dt_nu = dt_nu
        self.inv_dx2 = inv_dx2
        self.inv_dy2 = inv_dy2
        self.u_wall_count = u_wall_count
        self.v_wall_count = v_wall_count
        self.a_e = a_e
        self.a_w = a_w
        self.a_n = a_n
        self.a_s = a_s
        self.pressure_denominator = pressure_denominator
        self.fluid_center = fluid_center
        self.red_cells = red_cells
        self.black_cells = black_cells
        self.pressure_max_iterations = pressure_max_iterations
        self.pressure_residual_tolerance = pressure_residual_tolerance
        self.sor_omega = sor_omega
        self.apply_pressure_bc = apply_pressure_bc


    def backward_divergence(self, U_field, V_field):
        return (U_field[1:-1, 1:-1] - U_field[1:-1, :-2]) / self.dx + (V_field[1:-1, 1:-1] - V_field[:-2, 1:-1]) / self.dy

    def pressure_candidate(self, P_field, rhs):
        P_e = P_field[1:-1, 2:]
        P_w = P_field[1:-1, :-2]
        P_n = P_field[2:, 1:-1]
        P_s = P_field[:-2, 1:-1]
        numerator = self.a_e * P_e + self.a_w * P_w + self.a_n * P_n + self.a_s * P_s - rhs
        candidate = np.zeros_like(rhs)
        active = self.pressure_denominator > 0.0
        candidate[active] = numerator[active] / self.pressure_denominator[active]
        return candidate

    def poisson_residual(self, P_field, rhs):
        P_c = P_field[1:-1, 1:-1]
        P_e = P_field[1:-1, 2:]
        P_w = P_field[1:-1, :-2]
        P_n = P_field[2:, 1:-1]
        P_s = P_field[:-2, 1:-1]
        laplacian_P = self.a_e * (P_e - P_c) + self.a_w * (P_w - P_c) + self.a_n * (P_n - P_c) + self.a_s * (P_s - P_c)
        return laplacian_P - rhs

    def solve_pressure(self, P_field, rhs):
        P_inner = P_field[1:-1, 1:-1]
        for iteration in range(self.pressure_max_iterations):
            self.apply_pressure_bc(P_field)
            candidate = self.pressure_candidate(P_field, rhs)
            P_inner[self.red_cells] = (1.0 - self.sor_omega) * P_inner[self.red_cells] + self.sor_omega * candidate[self.red_cells]
            self.apply_pressure_bc(P_field)
            candidate = self.pressure_candidate(P_field, rhs)
            P_inner[self.black_cells] = (1.0 - self.sor_omega) * P_inner[self.black_cells] + self.sor_omega * candidate[self.black_cells]
            self.apply_pressure_bc(P_field)
            if (iteration + 1) % 10 == 0 or iteration == self.pressure_max_iterations - 1:
                residual = self.poisson_residual(P_field, rhs)
                max_residual = np.max(np.abs(residual[self.fluid_center]))
                if max_residual < self.pressure_residual_tolerance:
                    return (P_field, iteration + 1, max_residual)
        residual = self.poisson_residual(P_field, rhs)
        max_residual = np.max(np.abs(residual[self.fluid_center]))
        return (P_field, self.pressure_max_iterations, max_residual)

    def predict_velocity(self, U, V):
        U_center = U[1:-1, 1:-1]
        V_center = V[1:-1, 1:-1]
        V_at_u = 0.25 * (V[1:-1, 1:-1] + V[:-2, 1:-1] + V[1:-1, 2:] + V[:-2, 2:])
        U_at_v = 0.25 * (U[1:-1, 1:-1] + U[1:-1, :-2] + U[2:, 1:-1] + U[2:, :-2])
        dV_dx = np.where(U_at_v >= 0.0, (V_center - V[1:-1, :-2]) / self.dx, (V[1:-1, 2:] - V_center) / self.dx)
        dU_dx = np.where(U_center >= 0.0, (U_center - U[1:-1, :-2]) / self.dx, (U[1:-1, 2:] - U_center) / self.dx)
        dU_dy = np.where(V_at_u >= 0.0, (U_center - U[:-2, 1:-1]) / self.dy, (U[2:, 1:-1] - U_center) / self.dy)
        dV_dy = np.where(V_center >= 0.0, (V_center - V[:-2, 1:-1]) / self.dy, (V[2:, 1:-1] - V_center) / self.dy)
        laplacian_U = (U[1:-1, 2:] - 2.0 * U_center + U[1:-1, :-2]) * self.inv_dx2 + (U[2:, 1:-1] - 2.0 * U_center + U[:-2, 1:-1]) * self.inv_dy2
        laplacian_V = (V[1:-1, 2:] - 2.0 * V_center + V[1:-1, :-2]) * self.inv_dx2 + (V[2:, 1:-1] - 2.0 * V_center + V[:-2, 1:-1]) * self.inv_dy2
        laplacian_U -= self.u_wall_count * U_center * self.inv_dy2
        laplacian_V -= self.v_wall_count * V_center * self.inv_dx2
        advection_U = U_center * dU_dx + V_at_u * dU_dy
        advection_V = U_at_v * dV_dx + V_center * dV_dy
        U_star = U.copy()
        V_star = V.copy()
        U_star[1:-1, 1:-1] = U_center - self.dt * advection_U + self.dt_nu * laplacian_U
        V_star[1:-1, 1:-1] = V_center - self.dt * advection_V + self.dt_nu * laplacian_V
        return (U_star, V_star)
