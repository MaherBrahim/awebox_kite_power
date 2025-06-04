# Kalman Filter for Tether Reelout Acceleration Estimation using CasADi
# ---------------------------------------------------
# This implementation is based on the concepts from:
# M. Laaraiedh, "Implementation of Kalman Filter with Python Language", 2012.
# arXiv:1204.0375 [cs.CE] – https://arxiv.org/abs/1204.0375
#
# Adapted and extended to:
# - Use CasADi for symbolic matrix operations
# - Support a 3D state [length, velocity, acceleration]
# Author: Maher Brahim, 01.04.2025


import casadi as ca
import numpy as np
from scipy.signal import savgol_filter

def predict(x, P, A, Q):
    x_pred = ca.mtimes(A, x)
    P_pred = ca.mtimes(A, ca.mtimes(P, A.T)) + Q
    return x_pred, P_pred

def update(x_pred, P_pred, z, H, R):
    # the measurement residual on time step k
    y = z - ca.mtimes(H, x_pred)
    # the innovation covariance
    S = ca.mtimes(H, ca.mtimes(P_pred, H.T)) + R
    # the Kalman gain
    K = ca.mtimes(P_pred, ca.mtimes(H.T, ca.inv(S)))
    # the updated state estimate
    x_updated = x_pred + ca.mtimes(K, y)
    # the updated covariance matrix
    P_updated = ca.mtimes(ca.DM.eye(x_pred.size1()) - ca.mtimes(K, H), P_pred)
    likelihood = pdf_gauss(y, ca.DM.zeros(y.size1()), S)
    return x_updated, P_updated, K, likelihood

def pdf_gauss(x, mean, cov):
    n = x.size1()
    cov_inv = ca.inv(cov)
    exponent = 0.5 * ca.mtimes([x.T, cov_inv, x])
    denominator = ca.sqrt((2 * np.pi)**n * ca.det(cov))
    pdf = ca.exp(-exponent) / denominator
    return pdf

def kalman_filter_for_tether(time, l_meas, dl_meas, noise_params):
    n = len(time)
    dt_mean = np.diff(time).mean()
    # Initial state vector [length, velocity, acceleration] 
    x = ca.DM([l_meas[0], dl_meas[0], 0.0])

    # Initial covariance matrix with small uncertainties
    P = ca.DM.eye(3) * 1e-3

    # Initial state transition matrix
    A = ca.DM([[1, dt_mean, 0.5 * dt_mean**2],
               [0, 1, dt_mean],
               [0, 0, 1]])
    
    # Measurement matrix
    H = ca.DM([[1, 0, 0],
               [0, 1, 0]])

    # Process noise covariance matrix
    Q = ca.diag(ca.DM([1e-10, 1e-10, 1e1]))
   
    
    # Measurement noise covariance matrix using provided noise parameters
    R = ca.diag(ca.DM([noise_params['measurement_noise_length'], 
                        noise_params['measurement_noise_velocity']]))

    x_est = ca.DM.zeros((n, 3))
    x_est[0, :] = x.T

    for k in range(1, n):
        # Define the transition matrix
        dt = time[k] - time[k - 1]
        A[0, 1], A[0, 2], A[1, 2] = dt, 0.5 * dt**2, dt

        # --- Prediction Step ---
        x_pred, P_pred = predict(x, P, A, Q)
        # --- Update Step ---
        z = ca.DM([l_meas[k], dl_meas[k]])
        x, P, _, _ = update(x_pred, P_pred, z, H, R)

        x_est[k, :] = x.T

    return {
        'estimated_length': np.array(x_est[:, 0]).flatten(),
        'estimated_velocity': np.array(x_est[:, 1]).flatten(),
        'estimated_acceleration': np.array(x_est[:, 2]).flatten()
    }

def kalman_filter_derivation(time, y_meas, threshold=0.01):
    dt = np.diff(time).mean()
    n = len(time)

    A = ca.DM([[1, dt], [0, 1]])
    H = ca.DM([[1, 0]])

    # Use Savitzky-Golay filter to estimate smoothed signal
    y_smooth = savgol_filter(y_meas, window_length=7, polyorder=2)
    meas_residuals = y_meas - y_smooth
    R = ca.DM([[0.0001*np.var(meas_residuals)]])


    dy_with_savgol_filter = np.gradient(y_meas, time)
    dy_smooth = savgol_filter(dy_with_savgol_filter, window_length=11, polyorder=2)
    process_residuals = dy_with_savgol_filter - dy_smooth

    Q = ca.diag(ca.DM([1e-10, 1e-1]))

    dy_start = (y_meas[3] - y_meas[0])/3*dt
    x = ca.DM([y_meas[0], dy_start ])
    P = ca.DM.eye(2)* 1e-4

    estimated_y = np.zeros(n)
    estimated_dy = np.zeros(n)
    estimated_y[0] = y_meas[0]
    estimated_dy[0] = dy_start

    for k in range(1, n):
        dt = time[k] - time[k - 1]
        A[0, 1] = dt

        x_pred, P_pred = predict(x, P, A, Q)
        z = ca.DM([y_meas[k]])

        x_temp, P_temp, residual, S = update(x_pred, P_pred, z, H, R)
        likelihood = float(pdf_gauss(residual, ca.DM.zeros(residual.shape), S))
        x = x_pred

        if likelihood < threshold:
            x = x_pred
        else:
            x = x_temp
            P = P_temp

        estimated_y[k] = float(x[0])
        estimated_dy[k] = float(x[1])

    return estimated_y, estimated_dy

def forward_simulate(time, signal, order, initial_length, initial_velocity=0.0):
    """
    Propagate length (and velocity) using either velocity or acceleration.

    Parameters
    ----------
    time : measurement time [s].
    signal : If order=2: acceleration [m/s²]; if order=1: velocity [m/s].
    order : 1 to integrate velocity to length, 2 to integrate acceleration to velocity and length.
    initial_length : Tether length at time[0] [m].
    initial_velocity : Reelout velocity at time[0] [m/s] (only used if order=2).

    Returns
    -------
    If order=2:
        length_sim : Simulated tether length [m].
        velocity_sim : Simulated reelout velocity [m/s].
    If order=1:
        length_sim : Simulated tether length [m].
    """
    t = np.asarray(time, dtype=float)
    s = np.asarray(signal, dtype=float)
    N = t.size

    length_sim = np.zeros(N)
    length_sim[0] = initial_length

    if order == 2:
        velocity_sim = np.zeros(N)
        velocity_sim[0] = initial_velocity

        for k in range(1, N):
            dt = t[k] - t[k-1]
            a_prev = s[k-1]
            length_sim[k]   = length_sim[k-1]   + velocity_sim[k-1]*dt + 0.5*a_prev*dt**2
            velocity_sim[k] = velocity_sim[k-1] + a_prev*dt

        return length_sim, velocity_sim

    elif order == 1:
        for k in range(1, N):
            dt = t[k] - t[k-1]
            v_prev = s[k-1]
            length_sim[k] = length_sim[k-1] + v_prev*dt

        return length_sim

    else:
        raise ValueError("order must be 1 or 2")
