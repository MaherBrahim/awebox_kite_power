#!/usr/bin/python3
"""
some functions to process the measurements of the kitepower LEI kite. 
The measurements are taken from the kitepower data set, which was provied by kitepower.

:author: Maher Brahim
"""
# %%
# imports
import awebox as awe
import awebox.sim_kitepower_lei as sim
import casadi as ca
import awebox.opts.kite_data.kitepower_lei_data as kitepower_lei_data
import copy
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("module://matplotlib_inline.backend_inline")
# from matplotlib.collections import LineCollection
import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import numpy as np
import awebox.tools.print_operations as print_op
import json
import os
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from  plotting import plot_xy, plot_xyz, animate_3d_flight, is_gaussian_noise
from kalman_filter import  kalman_filter_for_tether, kalman_filter_derivation
from awebox.opts.kite_data.kitepower_lei_data import data_dict as data_dict_func

# Define path to measurements dataset
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(current_path, "..", "..", "Data", "DataShots"))
json_file = os.path.join(data_path,  "one_loop_meas_2025_1.json")

with open(json_file, "r") as f:
    data = json.load(f)


def savgol_derivative(time, values, window_length=10, poly_order=3):
    """
    Computes the first derivative using a Savitzky-Golay filter.
    """
    dt = np.mean(np.diff(time))
    dl_sg = savgol_filter(values, window_length, poly_order, deriv=1, delta=dt)
    return dl_sg


def derivative(time, values):
    grid = [time.tolist()] 
    values_list = values.tolist() 
    interpolation = ca.interpolant('interpolation', 'linear', grid, values_list)
    t_sym = ca.MX.sym('t')
    f_interp = interpolation(t_sym)

    # calulate the continous derivatives
    df_dt = ca.jacobian(f_interp, t_sym)
    f_deriv = ca.Function('f_deriv', [t_sym], [df_dt])

    # Calculation of the discrete derivatives
    derivative = np.array(f_deriv(time.tolist()))

    return  derivative

def discrete_derivative(time, position):
    time = np.asarray(time)
    position = np.asarray(position)
    n = len(time)
    derivative = np.zeros(n)
    derivative[0] = (position[1] - position[0]) / (time[1] - time[0])
    for i in range(1, n-1):
        derivative[i] = (position[i+1] - position[i-1]) / (time[i+1] - time[i-1])
    derivative[-1] = (position[-1] - position[-2]) / (time[-1] - time[-2])
    return derivative

def rotate_enu(wind_angle, east, north, up):
    angle = np.radians(wind_angle- 270)
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle),  np.cos(angle), 0],
                  [0,              0,             1]])
    enu = np.array([east, north, up])
    rotated_enu = np.dot(R, enu)
    return rotated_enu[0], rotated_enu[1], rotated_enu[2]

def remove_outliers(data, window_size, threshold):
    s = pd.Series(data)
    rm = s.rolling(window=window_size, center=True).median()
    deviation = np.abs(s - rm)
    s[deviation > threshold] = np.nan
    return s.values

def interpolate_data(filtered_data):
    s = pd.Series(filtered_data)
    x = np.arange(len(s))
    valid_mask = ~s.isna()
    x_valid = x[valid_mask]
    y_valid = s[valid_mask].values
    f = ca.interpolant('f', 'linear', [x_valid.tolist()], y_valid.tolist())
    return np.array([float(f(i)) for i in x])

def noise_estimation(time, length, velocity, window_length=21, polyorder=3):

    # Smooth signals
    length_smooth = savgol_filter(length, window_length, polyorder)
    velocity_smooth = savgol_filter(velocity, window_length, polyorder)

    # estimate acceleration from smoothed velocity
    acceleration_smooth = np.gradient(velocity, time)

    # Measurement noise variances
    r_length = np.var(length - length_smooth)
    r_velocity = np.var(velocity - velocity_smooth)

    # process noise variances
    acceleration_smooth_2 = savgol_filter(acceleration_smooth, window_length, polyorder)
    q_acceleration =  np.var(acceleration_smooth - acceleration_smooth_2)


    return {
        'length_smooth': length_smooth,
        'velocity_smooth': velocity_smooth,
        'acceleration_smooth': acceleration_smooth,
        'measurement_noise_length': r_length,
        'measurement_noise_velocity': r_velocity,
        'process_noise_acceleration': q_acceleration
    }

def check_uniform_dt(t, rtol=1e-9, atol=0.0):
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return True, np.nan
    dt = np.diff(t)
    uniform = np.allclose(dt, dt[0], rtol=rtol, atol=atol)
    return bool(uniform), (dt[0] if uniform else dt)

def common_dt_precision(t, max_decimals=12):
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return None, np.nan
    dt = np.diff(t)
    ref = dt[0]
    for d in range(max_decimals + 1):
        if not np.all(np.round(dt, d) == np.round(ref, d)):
            return d - 1, ref
    return max_decimals, ref


def get_variance(measurement, window_length, polyorder):
    """
    """
    y_meas = np.asarray(measurement, dtype=float)


    smoothed_meas = savgol_filter(y_meas, window_length, polyorder)
    residuals = y_meas - smoothed_meas

    variance = np.var(residuals, ddof=1)

    maxi = np.max(np.abs(y_meas))
    exponent = 0 if maxi == 0 else int(np.floor(np.log10(maxi)))
    weight_factor = 10.0 ** exponent
    
    return variance, weight_factor

def get_weighted_cov(measurements, window_length=11, polyorder=3):
    m = np.asarray(measurements, dtype=float)
    n_m = m[:,1].shape[0]
    cov = np.zeros((n_m, n_m))
    weighting_factors = np.zeros((n_m, n_m)) 
    for i in range(n_m):
        var_i, w_i = get_variance(m[i, :], window_length, polyorder)
        cov[i, i] = var_i # / (w_i ** 2)
        weighting_factors[i, i] = w_i
    return cov, weighting_factors 




if __name__ == "__main__":
    # Define the states und inputs from the measurements
    # time t
    time = np.array(data['time']) - data['time'][0]  
    # states
    upwind_direction_without_outliers = remove_outliers(data['ground_upwind_direction'], 50, 100)
    upwind_direction_filtered = interpolate_data(upwind_direction_without_outliers)
    upwind_direction_mean = np.mean(upwind_direction_filtered)
    upwind_direction_mean_vec = np.full(len(time), upwind_direction_mean)
    upwind_velocity_without_outliers = remove_outliers(data['ground_wind_velocity'], 50, 2)
    upwind_velocity_filtered = interpolate_data(upwind_velocity_without_outliers)

    x, y, z = np.array(
            [rotate_enu(a, e, n, u) for a, e, n, u in zip(
                upwind_direction_mean_vec,
                data['kite_pos_east'],
                data['kite_pos_north'],
                data['kite_height']
            )]).T

    v_x, v_y, v_z = np.array(
            [rotate_enu(a, e, n, u) for a, e, n, u in zip(
                upwind_direction_mean_vec,
                data['kite_est_vx'],
                data['kite_est_vy'],
                data['kite_est_vz']
            )]).T

    u_s = np.array(data['kite_actual_steering']) 
    u_d = np.array(data['kite_actual_depower'])
    l_t = np.array(data['ground_tether_length']) 
    dl_t = np.array(data['ground_tether_reelout_speed'])

    y_meas = ca.DM([x, y, z, v_x, v_y, v_z, u_s, u_d, l_t, dl_t])
    # inputs
    u_s_kf, du_s_kf  = kalman_filter_derivation(time, u_s)
    u_d_kf, du_d_kf = kalman_filter_derivation(time, u_d)
    noises = noise_estimation(time, l_t, dl_t)
    KF_results = kalman_filter_for_tether(time, l_t, dl_t, noises)
    ddl_t = KF_results['estimated_acceleration']
    
    u_meas = ca.DM([du_s_kf, du_d_kf, ddl_t]) 
    
    # define the initial states:
    x0 = y_meas[:, 0]
    z0 = 1.0

    # produce some plots 
    # ground wind velocity and upwind velocity filtered
    fig, ax = plot_xy(time, [data['ground_wind_velocity'],
                           upwind_velocity_filtered],
                            labels=['ground_wind_velocity', 
                            'upwind velocity filtered'], 
                            xlabel='time (s)', 
                            ylabel='velocity (m/s)', 
                            title='ground_wind_velocity over time')
    
    # measured and using Kalman Filter estimated tether length, velocity and acceleration 
    fig_lt, ax_lt = plot_xy(time, [KF_results['estimated_length'], l_t ], labels=['estimated l_t','measured l_t'], xlabel='time (s)', ylabel='l_t (m)', title='tether length')
    fig_dlt, ax_dlt = plot_xy(time, [KF_results['estimated_velocity'], dl_t ], labels=['estimated dl_t','measured dl_t'], xlabel='time (s)', ylabel='dl_t (m/s)', title='tether velocity')
    fig_ddlt, ax_ddlt = plot_xy(time, [KF_results['estimated_acceleration']], labels=['estimated ddl_t'], xlabel='time (s)', ylabel='ddl_t (m/s^2)', title='tether acceleration')

    # measured and using Kalman Filter estimated kite steering and depower
    fig00, ax00 = plot_xy(time, [du_s_kf], labels=['du_s using Kalman Filter'], xlabel='time (s)', ylabel='velocity ', title='du_s  over time')
    fig01, ax01 = plot_xy(time, [u_s, u_s_kf], labels=['u_s measured', 'u_s using Kalman Filter'], xlabel='time (s)', ylabel='u_s(%)', title='u_s  over time')
    fig02, ax02 = plot_xy(time, [du_d_kf], labels=['du_d using Kalman Filter'], xlabel='time (s)', ylabel='velocity ', title='du_d  over time')
    fig03, ax03 = plot_xy(time, [u_d, u_d_kf], labels=['u_d measured', 'u_d using Kalman Filter'], xlabel='time (s)', ylabel='u_d(%)', title='u_d  over time')

    # measured kite psition and velocity oriented in the wind direction
    fig_q, ax_q = plot_xyz( x, y, z, xlabel='X-pos', ylabel='Y-pos', zlabel='Z-pos', title='fligh path from measurment values (filtered)')
    fig_2d_q, ax_2d_q = plot_xy(time, [x, y, z], labels=['x', 'y', 'z'], xlabel='time (s)', ylabel='position ', title='kite position from measurment values (filtered)')
    fig_dq, ax_dq = plot_xyz( v_x, v_y, v_z, xlabel='X-vel', ylabel='Y-vel', zlabel='Z-vel', title='kite velocity from measurment values (filtered)')
    fig_2d_dq, ax_2d_dq = plot_xy(time, [v_x, v_y, v_z], labels=['v_x', 'v_y', 'v_z'], xlabel='time (s)', ylabel='velocity ', title='kite velocity from measurment values (filtered)')

    # Difference between the tether length and the distance of the kite to the groundstation 
    data_dict = data_dict_func()
    q_squares = [float(ca.mtimes(ca.DM([xi, yi, zi]).T, ca.DM([xi, yi, zi]))) for xi, yi, zi in zip(x, y, z)]
    l_t_square = (l_t + data_dict['geometry']['h_bridle'] + data_dict['geometry']['h_kite'])**2
    l_t_with_offset = l_t + data_dict['geometry']['h_bridle'] + data_dict['geometry']['h_kite'] 
    q_squares = np.array(q_squares)
    kite_distance =  data['kite_distance']
    kite_distance_kf, _ = kalman_filter_derivation(time, l_t_with_offset)
    offset = np.mean(kite_distance) - np.mean(kite_distance_kf)
    fig2d_kd, ax2d_kd = plot_xy(time, 
                                [l_t_with_offset , kite_distance, np.sqrt(q_squares), kite_distance - l_t_with_offset], 
                                labels=['l_t', 'kite distance to the GS measured', 'kite distance calculated from measurments', 'diff_measured'], 
                                xlabel='time (s)', 
                                ylabel='distance (m)', 
                                title=' diff between tether length (with h_kite and h_bridle) and kite position')
    
    fig2d_kd_offset, ax2d_kd_offset = plot_xy(time, 
                                [l_t_with_offset + offset , kite_distance, np.sqrt(q_squares), kite_distance - l_t_with_offset], 
                                labels=['l_t', 'kite distance to the GS measured', 'kite distance calculated from measurments', 'diff_measured'], 
                                xlabel='time (s)', 
                                ylabel='distance (m)', 
                                title=' diff between tether length (with h_kite, h_bridle and offset_value) and kite position')

    # fig2d_kd, ax2d_kd = plot_xy(np.array(l_t_with_offset)**2, 
    #                             [np.array(kite_distance)**2], 
    #                             labels=['l_t with offset^2'], 
    #                             xlabel='', 
    #                             ylabel='distance^2 (m)', 
    #                             title='tether length and kite position ')
    
    # animate_3d_flight([x_f, y_f, z_f], [], force_labels=[])
    weighted_cov = get_weighted_cov(y_meas, window_length=21, polyorder=3)
    print('weighted cov mtrix:', weighted_cov)
    plt.show()