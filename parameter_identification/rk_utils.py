import numpy as np
import casadi as ca




def rk4(f, x0, tf: float, n_steps: int = 1):
    # Compute time step from final time and number of steps
    dt = tf / n_steps

    # Create time vector
    t = np.linspace(0, tf, n_steps + 1)

    # Create storage for solution
    x = np.zeros((n_steps + 1, len(x0)))
    x[0, :] = x0

    # Runge-Kutta 4th order method
    for i in range(n_steps):
        k1 = f(x[i, :])
        k2 = f(x[i, :] + dt * k1 / 2)
        k3 = f(x[i, :] + dt * k2 / 2)
        k4 = f(x[i, :] + dt * k3)
        x[i + 1, :] = x[i, :] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4).full().flatten()

    x = x.tolist()
    return x, t


def rk4_on_timegrid(f, x0, t_grid: np.ndarray) -> list:

    x = np.zeros((len(t_grid) + 1, len(x0)))
    x[0, :] = x0

    # Runge-Kutta 4th order method
    for i in range(len(t_grid)):
        # TODO: multiple steps?
        dt = t_grid[i]
        k1 = f(x[i, :])
        k2 = f(x[i, :] + dt * k1 / 2)
        k3 = f(x[i, :] + dt * k2 / 2)
        k4 = f(x[i, :] + dt * k3)
        x[i + 1, :] = x[i, :] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4).full().flatten()

    x = x.tolist()
    return x

def generate_butcher_tableau_integral(n_s: int, irk_scheme):
    points = ca.collocation_points(n_s, irk_scheme)
    tau_root = np.array([0.0] + points)

    # Coefficients of the collocation equation
    C = np.zeros((n_s + 1, n_s + 1))
    # Coefficients of the continuity equation
    D = np.zeros((n_s + 1, 1))
    # Coefficients of the quadrature function
    B = np.zeros((n_s + 1, 1))

    # Construct polynomial basis
    for j in range(n_s + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        coeff = 1
        for r in range(n_s + 1):
            if not r == j:
                coeff = np.convolve(coeff, [1, -tau_root[r]])
                coeff = coeff / (tau_root[j] - tau_root[r])
        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = np.polyval(coeff, 1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(coeff)
        for r in range(n_s + 1):
            C[j, r] = np.polyval(pder, tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(coeff)
        B[j] = np.polyval(pint, 1.0)
    return B, C, D, tau_root


if __name__ == "__main__":
    # test RK tableaus
    for irk_scheme in ["legendre", "radau"]:
        n_s = 2
        B, C, D, tau_root = generate_butcher_tableau_integral(n_s, irk_scheme)
        print(f"Tableau for {n_s=} {irk_scheme} reads")
        print(f"{B=}\n{C=}\n{D=}\n{tau_root=}\n\n")
