#!/usr/bin/python3
"""
MPC-based closed loop simulation example for a single 3DOF kite system.

:author: Jochem De Schutter
:edited: Rachel Leuthold, Maher Brahim
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
from  kite_3D_plot import plot_xy, plot_xyz, animate_3d_flight, is_gaussian_noise
from kalman_filter import  kalman_filter_for_tether, kalman_filter_derivation
from awebox.opts.kite_data.kitepower_lei_data import data_dict as data_dict_func

# Define path to measurements dataset
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(current_path, "..", "..", "Data", "DataShots"))
json_file = os.path.join(data_path,  "data_for_power_cycle_2.json")

with open(json_file, "r") as f:
    data = json.load(f)

aggregated_data = {}
for plot in data.values():
    for key, value in plot.items():
        if key.endswith("_label"):
            base_key = key[:-6]
            if base_key in plot:
                label = value
                if label not in aggregated_data:
                    aggregated_data[label] = plot[base_key]
time = np.array(aggregated_data['time (s)']) - aggregated_data['time (s)'][0] 

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

    # Estimate acceleration from smoothed velocity
    acceleration_smooth = np.gradient(velocity_smooth, time)

    # Measurement noise variances
    r_length = np.var(length - length_smooth)
    r_velocity = np.var(velocity - velocity_smooth)

    # Process noise variance (acceleration)
    q_acceleration = 0.000001 * np.var(acceleration_smooth - np.mean(acceleration_smooth))


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

def run(plot_show_block=True, overwrite_options={}):

    # single kite with point-mass model
    options = {}
    options['user_options.system_model.architecture'] = {1: 0}
    options['user_options.kite_standard'] = kitepower_lei_data.data_dict()
    options['user_options.system_model.wing_type'] = 'LEI'
    options['user_options.system_model.kite_dof'] = 3

    # trajectory should be a single pumping cycle
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'lift_mode'
    windings = 4
    options['user_options.trajectory.lift_mode.windings'] = windings
    #options['model.system_bounds.theta.t_f'] =  [5.0, 15.0]

    # indicate desired environment
    options['params.wind.z_ref'] = 100.0
    options['params.wind.power_wind.exp_ref'] = 0.15
    options['user_options.wind.model'] = 'power'
    options['user_options.wind.u_ref'] = 6.
    


    # coefficient boundaries
    options['model.system_bounds.x.coeff'] =  [np.array([-1., 0.]), np.array([1., 1.])]
    options['model.system_bounds.u.dcoeff'] =  [np.array([-.08, -1]), np.array([.08, 1])]

    # indicate numerical nlp details
    # here: nlp discretization, with a zero-order-hold control parametrization, and
    # a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps
    # within ipopt.
    options['nlp.n_k'] = int(40/3 * windings)
    options['nlp.collocation.u_param'] = 'zoh'
    options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout' # 'simple' # 'single_reelout'
    options['solver.linear_solver'] = 'mumps'  # if HSL is installed, otherwise 'mumps'
    options['model.system_bounds.x.ddl_t'] = [-2.0, 2.0]
    options['model.system_bounds.theta.t_f'] = [0.0, windings*30.0]
    options['nlp.phase_fix_reelout'] = 0.7 

    options['model.model_bounds.acceleration.include']  = False
    options['model.model_bounds.aero_validity.include']  = False
    options['model.model_bounds.tether_stress.include']  = False
    # (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
    # note: this may result in slightly slower solution timings
    options['nlp.compile_subfunctions'] = False

    # initialization
    options['solver.initialization.shape'] = 'lemniscate'
    options['solver.initialization.lemniscate.az_width'] = 20*np.pi/180.
    options['solver.initialization.lemniscate.el_width'] = 8*np.pi/180.
    options['solver.initialization.inclination_deg'] = 30.
    options['solver.initialization.groundspeed'] = 20.
    options['solver.initialization.theta.diam_t'] = 5e-3
    options['solver.initialization.l_t'] = 300.0
    options['solver.max_iter_hippo'] = 1000
    options['solver.max_iter'] = 1000
    options['visualization.cosmetics.plot_ref'] = False

    options['model.tether.control_var'] = 'ddl_t'

    # initialize and optimize trial
    # build and optimize the NLP (trial)
    trial = awe.Trial(options, 'Kitepower_LEI')
    trial.build()
    trial.optimize(final_homotopy_step = 'power')  # 'initial_guess', 'initial', 'fictitious', 'power', 'final'
    trial.plot(['isometric', 'states', 'controls', 'constraints'])

    plot_dict = trial.visualization.plot_dict
    outputs = plot_dict['outputs']
    time = plot_dict['time_grids']['ip']
    x = plot_dict['x']
    


    plt.show(block=plot_show_block)

    # set-up open-loop simulation
    N_sim = 200  # closed-loop simulation steps
    ts = 0.1 # sampling time

    # simulation options
    options['sim.number_of_finite_elements'] = 20 # integrator steps within one sampling time
    options['sim.sys_params'] = copy.deepcopy(trial.options['solver']['initialization']['sys_params_num'])

    # reduce average wind speed
    #options['sim.sys_params']['wind']['u_ref'] = 1.0*options['sim.sys_params']['wind']['u_ref']

    # make simulator
    open_loop_sim = sim.Simulation(trial, 'open_loop', ts, options)
    # define initial state and inputs
    x, y, z = rotate_enu(aggregated_data['ground_upwind_direction (deg)'][0], aggregated_data['kite_pos_east (m)'][0], aggregated_data['kite_pos_north (m)'][0], aggregated_data['kite_height (m)'][0])
    v_x, v_y, v_z = rotate_enu(aggregated_data['ground_upwind_direction (deg)'][0], aggregated_data['kite_est_vx (m/s)'][0], aggregated_data['kite_est_vy (m/s)'][0], aggregated_data['kite_est_vz (m/s)'][0])
    u_s = np.array(aggregated_data['kite_actual_steering (%)']) 
    u_d = np.array(aggregated_data['kite_actual_depower (%)'])
    l_t = np.array(aggregated_data['ground_tether_length (m)']) 
    dl_t = np.array(aggregated_data['ground_tether_reelout_speed (m/s)'])
    # calculate the acceleration ddl_t using kalman filter
    # Estimation of the noise parameters from the measurement data of dl_t and l_t
    noise_estimation = noise_estimation(time, l_t, dl_t)
    # Estimate the acceleration ddl_t using the KF
    KF_results = kalman_filter_for_tether(time, l_t, dl_t, noise_estimation)
    l_t_filtered = KF_results['estimated_length']
    dl_t_filtered = KF_results['estimated_velocity']
    ddl_t_filtered = KF_results['estimated_acceleration']

    # Derivation of u_s and u_d using KF
    u_s_filtered, du_s_filtered = kalman_filter_derivation(time, u_s)
    u_d_filtered, du_d_filtered = kalman_filter_derivation(time, u_d)


    X0 = ca.DM([x, y, z, v_x, v_y, v_z, u_s_filtered[0]/100, u_d_filtered[0]/100, l_t_filtered[0], dl_t_filtered[0]])
    
    
    zeros_dm = np.zeros(len(time))
    u_meas_list = []
    u_meas_list.append(zeros_dm)
    u_meas_list.append(zeros_dm)
    u_meas_list.append(zeros_dm)
    u_meas_list.append(du_s_filtered)
    u_meas_list.append(du_d_filtered)
    u_meas_list.append(ddl_t_filtered)
    u_meas_vert = ca.horzcat(*u_meas_list)

    # make simulator
    open_loop_sim = sim.Simulation(trial, 'open_loop', ts, options)
    # open_loop_sim.run(N_sim)
    # open_loop_sim.plot(['isometric','states'])
    # plt.show(block=plot_show_block)
 
    open_loop_sim.run(N_sim, x0=X0, u_sim=u_meas_vert, time=time)
    open_loop_sim.plot(['isometric','states'])
    plt.show(block=plot_show_block)

    return open_loop_sim

# %% 
# Main
if __name__ == "__main__":
    data_dict = data_dict_func()
    dl_t = np.array(aggregated_data['ground_tether_reelout_speed (m/s)'])
    l_t = np.array(aggregated_data['ground_tether_length (m)']) 
    results = noise_estimation(time, l_t, dl_t)
    x, y, z = np.array(
        [rotate_enu(a, e, n, u) for a, e, n, u in zip(
            aggregated_data['ground_upwind_direction (deg)'],
            aggregated_data['kite_pos_east (m)'],
            aggregated_data['kite_pos_north (m)'],
            aggregated_data['kite_height (m)']
        )]).T



    upwind_direction_without_outliers = remove_outliers(aggregated_data['ground_upwind_direction (deg)'], 50, 100)
    upwind_direction_filtered = interpolate_data(upwind_direction_without_outliers)
    upwind_direction_mean = np.mean(upwind_direction_filtered)
    N = len(time)
    upwind_direction_filtered_mean = np.full(N, upwind_direction_mean)
    
    # %%  
    # filtered flight path
    fig3, ax3 = plot_xy(time, [aggregated_data['ground_upwind_direction (deg)'], upwind_direction_filtered ], labels=['ground_upwind_direction', 'upwind_direction_filtered'], xlabel='time (s)', ylabel='ground_upwind_direction (deg)', title='ground upwind direction over time')
    fig3d_1, ax3d_1 = plot_xyz( aggregated_data['kite_pos_east (m)'], aggregated_data['kite_pos_north (m)'], aggregated_data['kite_height (m)'], xlabel='X-pos', ylabel='Y-pos', zlabel='Z-pos', title='fligh path from measurment values')
    fig3d_2, ax3d_2 = plot_xyz( x, y, z, xlabel='X-pos', ylabel='Y-pos', zlabel='Z-pos', title='fligh path from measurment values')
    x_f, y_f, z_f = np.array(
        [rotate_enu(a, e, n, u) for a, e, n, u in zip(
            upwind_direction_filtered_mean,
            aggregated_data['kite_pos_east (m)'],
            aggregated_data['kite_pos_north (m)'],
            aggregated_data['kite_height (m)']
        )]).T

    fig3d_3, ax3d_3 = plot_xyz( x_f, y_f, z_f, xlabel='X-pos', ylabel='Y-pos', zlabel='Z-pos', title='fligh path from measurment values (filtered)')

    plt.show()
    
    # %% 
    # Difference between the tether length and the distance of the kite to the groundstation 
    q_squares = [float(ca.mtimes(ca.DM([xi, yi, zi]).T, ca.DM([xi, yi, zi]))) for xi, yi, zi in zip(x, y, z)]
    l_t_square = (l_t + data_dict['geometry']['h_bridle'] + data_dict['geometry']['h_kite'])**2
    l_t_with_offset = l_t + data_dict['geometry']['h_bridle'] + data_dict['geometry']['h_kite'] 
    q_squares = np.array(q_squares)
    kite_distance =  aggregated_data['kite_distance (m)']
    kite_distance_kf, _ = kalman_filter_derivation(time, kite_distance)
    #offset = np.mean(kite_distance) - np.mean(l_t_with_offset)
    fig2d_kd, ax2d_kd = plot_xy(time, 
                                [l_t_with_offset , kite_distance, np.sqrt(q_squares), kite_distance - l_t_with_offset], 
                                labels=['l_t', 'kite distance to the GS measured', 'kite distance calculated from measurments', 'diff_measured'], 
                                xlabel='time (s)', 
                                ylabel='distance (m)', 
                                title=' diff between tether length (with h_kite and h_bridle) and kite position')

    fig2d_kd, ax2d_kd = plot_xy(np.array(l_t_with_offset)**2, 
                                [np.array(kite_distance)**2], 
                                labels=['l_t with offset^2'], 
                                xlabel='', 
                                ylabel='distance^2 (m)', 
                                title='tether length and kite position ')
    
    # animate_3d_flight([x_f, y_f, z_f], [], force_labels=[])

    plt.show()
    
    # diff = (np.sqrt(q_squares)  - (l_t + data_dict['geometry']['h_bridle'] + data_dict['geometry']['h_kite']))
    # 
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # 
    # points = np.array([kite_distance, l_t_with_offset]).T.reshape(-1,1,2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # norm = plt.Normalize(0, len(kite_distance)-1)
    # lc = LineCollection(segments, cmap='viridis', norm=norm)
    # lc.set_array(np.arange(len(kite_distance)))
    # ax1.add_collection(lc)
    # # ax1.set_xlim(min(q_squares), max(q_squares))
    # # ax1.set_ylim(min(l_t_square), max(l_t_square))
    # ax1.set_xlabel('q_norm')
    # ax1.set_ylabel('l_t_with_offset')
    # ax1.set_title('Visualization')
    # ax1.grid(True)
    # plt.colorbar(lc, ax=ax1)
    # ax2.plot(time, kite_distance - l_t_with_offset)
    # ax2.set_xlabel('time')
    # ax2.set_ylabel('q- l_t')
    # ax2.set_title('Difference')
    # ax2.grid(True)
    # 
    # plt.tight_layout()

    # q_squares_with_filter = [
    # float(ca.mtimes(ca.DM([xi, yi, zi]).T, ca.DM([xi, yi, zi]))) for xi, yi, zi in zip(x_f, y_f, z_f)]
    # q_squares_wf = np.array(q_squares_with_filter)
    # 
    # diff = (np.sqrt(q_squares_wf) - (l_t + data_dict['geometry']['h_bridle'] + data_dict['geometry']['h_kite']) )
    # 
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # 
    # points = np.array([q_squares_wf, l_t_square]).T.reshape(-1,1,2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # norm = plt.Normalize(0, len(q_squares_wf)-1)
    # lc = LineCollection(segments, cmap='viridis', norm=norm)
    # lc.set_array(np.arange(len(q_squares_wf)))
    # ax1.add_collection(lc)
    # ax1.set_xlim(min(q_squares_wf), max(q_squares_wf))
    # ax1.set_ylim(min(l_t_square), max(l_t_square))
    # ax1.set_xlabel('q_square')
    # ax1.set_ylabel('l_t^2')
    # ax1.set_title('Visualization (filtered measurments)')
    # ax1.grid(True)
    # plt.colorbar(lc, ax=ax1)
    # ax2.plot(time, diff)
    # ax2.set_xlabel('time')
    # ax2.set_ylabel('q- l_t')
    # ax2.set_title('Difference')
    # ax2.grid(True)

    # plt.tight_layout()

    

    # %%
    # Kalman Filtering

    KF_results = kalman_filter_for_tether(time, l_t, dl_t, results)
    fig_lt, ax_lt = plot_xy(time, [KF_results['estimated_length'], l_t ], labels=['estimated l_t','measured l_t'], xlabel='time (s)', ylabel='l_t (m)', title='tether length')
    fig_dlt, ax_dlt = plot_xy(time, [KF_results['estimated_velocity'], dl_t ], labels=['estimated dl_t','measured dl_t'], xlabel='time (s)', ylabel='dl_t (m/s)', title='tether velocity')
    fig_ddlt, ax_ddlt = plot_xy(time, [KF_results['estimated_acceleration']], labels=['estimated ddl_t'], xlabel='time (s)', ylabel='ddl_t (m/s^2)', title='tether acceleration')

    u_s = np.array(aggregated_data['kite_actual_steering (%)'])
    u_d = np.array(aggregated_data['kite_actual_depower (%)'])


    u_s_kf, du_s_kf = kalman_filter_derivation(time, u_s)
    u_d_kf, du_d_kf = kalman_filter_derivation(time, u_d)
    du_s = savgol_derivative(time, u_s) 
    du_d = savgol_derivative(time, u_d) 
    upwind_kf_filter,_ = kalman_filter_derivation(time, aggregated_data['ground_upwind_direction (deg)'], .1)
    fig00, ax00 = plot_xy(time, [du_s, du_s_kf], labels=['du_s using Savgol', 'du_s using Kalman Filter'], xlabel='time (s)', ylabel='velocity ', title='du_s  over time')
    fig01, ax01 = plot_xy(time, [u_s, u_s_kf], labels=['u_s measured', 'u_s using Kalman Filter'], xlabel='time (s)', ylabel='u_s(%)', title='u_s  over time')
    fig02, ax02 = plot_xy(time, [du_d, du_d_kf], labels=['du_d using Savgol', 'du_d using Kalman Filter'], xlabel='time (s)', ylabel='velocity ', title='du_d  over time')
    fig03, ax03 = plot_xy(time, [u_d, u_d_kf], labels=['u_d measured', 'u_d using Kalman Filter'], xlabel='time (s)', ylabel='u_d(%)', title='u_d  over time')
    fig02, ax02 = plot_xy(time, [aggregated_data['ground_upwind_direction (deg)'], upwind_kf_filter], labels=['ground_upwind_direction  measured', 'ground_upwind_direction  using Kalman Filter'], xlabel='time (s)', ylabel='velocity (m/s)', title='du_s  over time')
    plt.show()
    # %%
    # calculate the dts in time 
    dt = check_uniform_dt(time, rtol=1e-9, atol=1e-6,)
    print ('Time intervals between the individual points in time: ', dt)
    dec, ref = common_dt_precision(time)
    print(f"dt is equal up to {dec} decimal places, value = {ref}\n")

    # %%
    # Check whether a Gaussian noise pattern exists for the given data 
    is_gaussian_noise(dl_t)
    is_gaussian_noise(upwind_direction_without_outliers)
    # %%
    # simulation run
    # open_loop_sim = run()
# %%
