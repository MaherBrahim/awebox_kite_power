import casadi as ca, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awebox as awe
from awebox.mdl.architecture import Architecture
import awebox.tools.struct_operations as struct_op
import awebox.opts.kite_data.kitepower_lei_data as kite_data
from wrapper_for_sysid import generate_implicit_dae_F, get_bounds, flatten_group_bounds, get_scaled_bounds, get_scaled_vars, get_reverse_rescaled_vars
from rk_utils import generate_butcher_tableau_integral
from scipy.signal import savgol_filter
from plotting import plot_xy, plot_xyz, plot_xy_mixed,plot_3d_mixed,   animate_3d_flight, is_gaussian_noise
from kalman_filter import  kalman_filter_for_tether, kalman_filter_derivation
from  measurement_processing import rotate_enu, remove_outliers, interpolate_data, noise_estimation, get_weighted_cov
from awebox.opts.kite_data.kitepower_lei_data import data_dict as data_dict_func
import awebox.mdl.model as mdl
import awebox.mdl.architecture as archi
import awebox.opts.options as opts
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import os, sys, pathlib, ctypes, json

dll_dir = pathlib.Path(r"C:\Users\maher\OneDrive\Desktop\Masterarbeit\Code2.0\awebox_kite_power\toolchain\bin")

# Windows only finds the dependency DLLs if the directory is known
if sys.version_info >= (3, 8):
    os.add_dll_directory(str(dll_dir))


# Jetzt die HSL‑DLL laden (vollem Pfad!)
ctypes.CDLL(str(dll_dir / "libhsl.dll"))
print("libhsl.dll erfolgreich geladen")

def setup_model():
    """
    Set up the model and options for the kitepower system.
    """
    upwind_velocity_without_outliers = remove_outliers(data['ground_wind_velocity'], 50, 2)
    upwind_velocity_filtered = interpolate_data(upwind_velocity_without_outliers)
    upwind_velocity_mean = np.mean(upwind_velocity_filtered)
    # Load the options
    options_seed = {} 
    options_seed['user_options.wind.u_ref'] = upwind_velocity_mean
    options_seed = ampyx_ap2_settings.set_kitepower_lei_settings(options_seed)
    options = opts.Options()
    options.fill_in_seed(options_seed)

    # Create the model
    model = mdl.Model()
    architecture = archi.Architecture(options['user_options']['system_model']['architecture'])
    options.build(architecture)
    model.build(options['model'], architecture)

    return model, options

# Load the kite geometry parameters dictionary
data_dict = data_dict_func()

# Define path to measurements dataset
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(current_path, "..", "..", "Data", "DataShots"))
json_file = os.path.join(data_path,  "one_loop_meas_2025_2.json")

with open(json_file, "r") as f:
    data = json.load(f)


# # define the bounds 
# lb, ub = get_bounds()
# lb_x, ub_x = flatten_group_bounds(lb, ub, 'x')
# lb_u, ub_u = flatten_group_bounds(lb, ub, 'u')
# lb_z, ub_z = flatten_group_bounds(lb, ub, 'z')
# lb_p, ub_p = flatten_group_bounds(lb, ub, 'p')

def tether_constraints(model, x_scaled):

    x_pysical = get_reverse_rescaled_vars(model, x=x_scaled)
    # pick the tether lenghth and reelout speed 
    l_t = x_pysical[8]
    dl_t = x_pysical[9]
    # pick the kite position und velocity
    q = x_pysical[0:3]
    dq = x_pysical[3:6]

    # define the constraints:
    c = 0.5 * (q.T @ q - l_t**2)
    c_dot = q.T  @ dq - dl_t * l_t

    return ca.vertcat(c, c_dot)

def setup_collocation(n_s:int, N_fe, t_meas: np.ndarray, y_meas: np.ndarray, 
                      u_meas: np.ndarray, X0: np.ndarray, Z0: np.ndarray, 
                       W_y: np.ndarray, W_theta: np.ndarray, theta_hat: np.ndarray):
    
    # define the bucher tableau coefficients:
    B, C, D, _ = generate_butcher_tableau_integral(n_s, "radau")


    N = y_meas.shape[1] - 1

    if  N < 2:
        raise ValueError("Need at least two time stamps for collocation")
    dt = t_meas[1] - t_meas[0]

    # Time horizon
    T = dt * (N)

    # discretization of the time horizon
    h = dt / N_fe

    # define the scaled bounds
    lb_scaled, ub_scaled = get_scaled_bounds(model)
    lb_x, ub_x = flatten_group_bounds(lb_scaled, ub_scaled, "x")
    lb_u, ub_u = flatten_group_bounds(lb_scaled, ub_scaled, "u")
    lb_z, ub_z = flatten_group_bounds(lb_scaled, ub_scaled, "z")
    lb_p, ub_p = flatten_group_bounds(lb_scaled, ub_scaled, "p")

    # define the penalty weights for slack variables
    W_s_dae = 1e3
    W_s_tether = 1e1   

    # Continuous time dynamics
    n_param = 0
    params_dict = {}
    params_dict['geometry'] = {}
    params_dict['geometry']['K_s,D'] = [1]
    n_param += 1
    params_dict['geometry']['c_s'] = [1]
    n_param += 1
    
    F_dae =  generate_implicit_dae_F(n_param, params_dict)
    

    # states and controls
    nx = 10
    nz = 1
    nu = 3
    x = ca.SX.sym('x', nx)
    z = ca.SX.sym('z', nz)
    u = ca.SX.sym('u', nu)
    x0 = ca.SX.sym('x0', nx)
    z0 = ca.SX.sym('x0', nz)


    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    objective = 0
    g = []
    lbg = []
    ubg = []

    # for plotting x, z and theta
    x_plot = []
    z_plot = []
    theta_plot = []
    slack_tether_plot = []
    slack_dae_plot = []

    # define the initial conditions for the states
    Xk = ca.SX.sym('X0', nx)
    w.append(Xk)
    #lbw.append(X0[0:6])
    lbw.append(lb_x[0:10])
    #lbw.append(lb_x[9:10])
    #ubw.append(ub_x[0:8])
    #ubw.append(X0[0:6])
    ubw.append(ub_x[0:10])
    w0.append(X0)
    x_plot.append(Xk)


    # define the initial conditions for the algebraic variables
    # Zk = ca.SX.sym('Z0', nz)
    # w.append(Zk)
    # lbw.append(lb_z)
    # ubw.append(ub_z)
    # w0.append(Z0)
    # z_plot.append(Zk)

    # initialize the slack variables
    # slack_tether = ca.SX.sym('slack_tether_0', 2)
    # w.append(slack_tether)
    # lbw.append(ca.DM.zeros(2))
    # ubw.append(np.inf*ca.DM.ones(2))
    # w0.append(ca.DM.zeros(2))
    # slack_tether_plot.append(slack_tether)

    # Enforce tether constraints at start
    g.append(tether_constraints(model, Xk))
    lbg.append(ca.DM.zeros(2))
    ubg.append(ca.DM.zeros(2))


    # g.append(tether_constraints(model, Xk) + slack_tether)
    # lbg.append(ca.DM.zeros(2))
    # ubg.append(np.inf*ca.DM.ones(2))
# 
    #define the initial conditions for the parameters
    theta = ca.SX.sym('theta', n_param) 
    w.append(theta)
    lbw.append(lb_p)
    ubw.append(ub_p)
    w0.append(theta_hat)
    theta_plot.append(theta)

    

    for k in range(0,N):
        # Loop over integration steps / finite elements
        for i_fe in range(N_fe):
            Xk_end = D[0] * Xk
            # Zk_end = D[0] * Zk
            # State at collocation points
            Xc = [] 
            Zc = []
            for j in range(n_s):
                Xkj = ca.SX.sym(f'X_{k}_{i_fe}_{j}', nx)
                Zkj = ca.SX.sym(f'Z_{k}_{i_fe}_{j}', nz)
                Xc.append(Xkj)
                Zc.append(Zkj)
                w.append(Xkj)
                w.append(Zkj)
                'TODO: add bounds for all states (done)'
                lbw.append(lb_x)
                ubw.append(ub_x)
                lbw.append(lb_z)
                ubw.append(ub_z)
                w0.append(y_meas[:, k])  
                w0.append(Z0)
            
            

            # Loop over collocation points
            for j in range(1, n_s + 1):
                # Expression for the state derivative at the collocation point
                xp = C[0, j] * Xk
                # zp = C[0, j] * Zk
                for r in range(n_s):
                    xp = xp + C[r + 1, j] * Xc[r]
                    x_dot = xp/h
                    #zp = zp + C[r + 1, j] * Zc[r]

                # Model DAE equations
                f_dae = F_dae(x_dot, Xc[j - 1], u_meas[:, k], Zc[j - 1], theta)

                # Add contribution to the end state 
                Xk_end = Xk_end + D[j] * Xc[j - 1]

            Zk_end = Zc[-1]    
            
            

            

            Xk = ca.SX.sym(f'X_{k+1}', nx)
            Zk = ca.SX.sym(f'Z_{k+1}', nz)
            w.append(Xk)
            w.append(Zk)
            lbw.append(lb_x)
            ubw.append(ub_x)
            lbw.append(lb_z)
            ubw.append(ub_z)
            w0.append(X0)
            w0.append(Z0) 
            x_plot.append(Xk)
            z_plot.append(Zk)



            # Add equality constraint
            g.append(Xk - Xk_end)
            lbg.append(np.zeros((nx,)))
            ubg.append(np.zeros((nx,)))

            # Add equality constraint for the algebraic variables 
            g.append(Zk - Zk_end)
            lbg.append(np.zeros((nz,)))
            ubg.append(np.zeros((nz,)))


            # slack_dae = ca.SX.sym(f'slack_dae_{k}', nx + nz)
            # w.append(slack_dae)
            # lbw.append(ca.DM.zeros(nx + nz))
            # ubw.append(ca.inf * ca.DM.ones(nx + nz))
            # w0.append(ca.DM.zeros(nx + nz))
            # slack_dae_plot.append(slack_dae)

            # Collocation equations
            g.append(f_dae)
            lbg.append(ca.DM.zeros(nx + nz))
            ubg.append(ca.DM.zeros(nx + nz))

            # g.append(f_dae + slack_dae)
            # lbg.append(ca.DM.zeros(nx + nz))
            # ubg.append(ca.inf * ca.DM.ones(nx + nz))

            #dae_slack_cost = W_s_dae * (slack_dae.T @ slack_dae)
            #dae_slack_cost = W_s_dae * ca.sum1(slack_dae)

            #calculate slack tether variables
            # lack_tether = ca.SX.sym('slack_tether', 2)
            # .append(slack_tether)
            # bw.append(ca.DM.zeros(2))
            # bw.append(np.inf*ca.DM.ones(2))
            # 0.append(ca.DM.zeros(2))
            # lack_tether_plot.append(slack_tether)

            # Tether constraints at node k
            # k = tether_constraints(model, Xk)
            # .append(Ck - slack_tether)
            # bg.append(-ca.inf * ca.DM.ones(2))
            # bg.append(ca.DM.zeros(2))

            # g.append(Ck + slack_tether)
            # lbg.append(ca.DM.zeros(2))
            # ubg.append(ca.inf * ca.DM.ones(2))

            #tether_slack_cost = (slack_tether.T @ W_s_tether  @ slack_tether)
            # tether_slack_cost = W_s_tether * ca.sum1(slack_tether)
                
            # define the objective function 
            objective +=  ((((y_meas[:, k+1] - Xk_end).T @ W_y @ (y_meas[:, k+1] - Xk_end)) )  +  
                           (theta-theta_hat).T @ W_theta @ (theta-theta_hat) )
        
    
    
    g_col = [ca.reshape(Gi, Gi.numel(), 1) for Gi in g]
    #print(len(g_col))
    # Concatenate vectors
    w = ca.vertcat(*w)
    g = ca.vertcat(*g_col)
    w0 = ca.vertcat(*w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = ca.vertcat(*lbg)
    ubg = ca.vertcat(*ubg)
    x_plot = ca.horzcat(*x_plot)
    z_plot = ca.horzcat(*z_plot)
    theta_plot = ca.horzcat(*theta_plot)
    slack_tether_plot = ca.horzcat(*slack_tether_plot)
    slack_dae_plot = ca.horzcat(*slack_dae_plot)



    nlp = {
        'f': objective,
        'x': w,
        'z' : z,
        'g': g,
        'p': theta,
        'w': w,
        'w0': w0,
        'lbw': lbw,
        'ubw': ubw,
        'lbg': lbg,
        'ubg': ubg
    }
    plt_data = {
        'x_plot': x_plot,
        'z_plot': z_plot,
        'theta_plot':theta_plot,
        'slack_tether_plot': slack_tether_plot,
        'slack_dae_plot': slack_dae_plot,
    }
    casadi_nlp = {'f': objective, 'x': w, 'g': g}
    return nlp, casadi_nlp, plt_data


def collocation_for_LSP (n_s, N_fe, t_meas, y_meas, u_meas,x0, z0, W_y, W_theta,theta_hat):

    
    # setup the collocation problem
    nlp, casadi_nlp, plt_data  = setup_collocation(n_s, N_fe, t_meas, y_meas, u_meas, x0, z0, W_y, W_theta, theta_hat)

    
    opts = {
    "ipopt": {
        "linear_solver": "ma57",                     # MA27 wirklich aktivieren
        "hsllib": str(dll_dir / "libhsl.dll"),
        }
    }
    solver = ca.nlpsol('solver', 'ipopt', casadi_nlp, opts)


    return nlp, solver, plt_data


if __name__ == "__main__":
    # Load the model and options
    model, options = setup_model()
    # Define the states und inputs from the measurements
    # time t
    t_meas_main = np.array(data['time']) - data['time'][0]  
    # states
    upwind_direction_without_outliers = remove_outliers(data['ground_upwind_direction'], 50, 100)
    upwind_direction_filtered = interpolate_data(upwind_direction_without_outliers)
    upwind_direction_mean = np.mean(upwind_direction_filtered)
    upwind_direction_mean_vec = np.full(len(t_meas_main), upwind_direction_mean)
    upwind_velocity_without_outliers = remove_outliers(data['ground_wind_velocity'], 50, 2)
    upwind_velocity_filtered = interpolate_data(upwind_velocity_without_outliers)
    # fig, ax = plot_xy(t, [data['ground_wind_velocity'],
    #                        upwind_velocity_filtered],
    #                         labels=['ground_wind_velocity', 
    #                         'upwind velocity filtered'], 
    #                         xlabel='time (s)', 
    #                         ylabel='velocity (m/s)', 
    #                         title='ground_wind_velocity over time')

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



    l_t_with_offset = l_t + data_dict['geometry']['h_bridle'] + data_dict['geometry']['h_kite']
    kite_distance =  data['kite_distance']
    kite_distance_kf, _ = kalman_filter_derivation(t_meas_main, l_t_with_offset)
    offset = []
    # for i in range(len(kite_distance)-1):
    #     l_t_with_offset[i] += (kite_distance[i] - kite_distance_kf[i])

    offset = np.mean(kite_distance) - np.mean(kite_distance_kf) 
    l_t_with_offset += offset


    y_meas_main = ca.DM([x, y, z, v_x, v_y, v_z, u_s/100, u_d/100, l_t_with_offset, dl_t])
    


    # inputs
    u_s_kf, du_s_kf  = kalman_filter_derivation(t_meas_main, u_s)
    u_d_kf, du_d_kf = kalman_filter_derivation(t_meas_main, u_d)
    noises = noise_estimation(t_meas_main, l_t, dl_t)
    KF_results = kalman_filter_for_tether(t_meas_main, l_t, dl_t, noises)
    ddl_t = KF_results['estimated_acceleration']
    du_s_without_outliers = remove_outliers(du_s_kf, 5, 10)
    du_s_filtered = interpolate_data(du_s_without_outliers)
    u_meas_main = ca.DM([du_s_filtered, du_d_kf, ddl_t]) 

    
    # scale the measurements and the inputs
    y_meas_main_scaled = ca.DM.zeros(y_meas_main.shape)
    u_meas_main_scaled = ca.DM.zeros(u_meas_main.shape)
    for i in range(y_meas_main.shape[0]):
        y_meas_main_scaled[:, i] = get_scaled_vars(model, x=y_meas_main[:, i])
    for i in range(u_meas_main.shape[0]):
        u_meas_main_scaled[:, i] = get_scaled_vars(model, u=u_meas_main[:, i])

    # print('y_meas_main_scaled:', y_meas_main_scaled)
    # print('u_meas_main_scaled:', u_meas_main_scaled)

    
    # define the initial states:
    X0_main = y_meas_main[:, 0]
    Z0_main = ca.DM([1.0])

    X0_main_scaled, Z0_main_scaled = get_scaled_vars(model, x=X0_main, z=Z0_main)
    # print('X0_main_scaled:', X0_main_scaled)
    # print('Z0_main_scaled:', Z0_main_scaled)

    W_y, weighting_mat = get_weighted_cov(y_meas_main, window_length=21, polyorder=3)
    W_theta = np.zeros((2, 2))
    W_theta[0,0] = 1e-3
    W_theta[1,1] = 1e-3

    theta_hat = ca.DM([0.6, 2.5])
    
    Nm = 13
    N= (Nm)-1
    # define the number of collocation points and the number of finite elements
    n_s= 1
    N_fe= 1



    # nlp, casadi_nlp = setup_collocation(N, n_s, N_fe, y_meas, u_meas, x0, Z0_main, W_y, W_theta, theta_hat)

    
    # Call the collocation function
    nlp, solver, plt_data = collocation_for_LSP(n_s, N_fe, t_meas_main[:Nm], y_meas_main_scaled[:,:Nm], u_meas_main_scaled[:,:Nm], X0_main_scaled, Z0_main_scaled, W_y, W_theta, theta_hat)
    #print(nlp['w0'])
    # Solve the collocation problem
    sol = solver(x0=nlp['w0'],
                 lbx=nlp['lbw'],
                 ubx=nlp['ubw'],
                 lbg=nlp['lbg'],
                 ubg=nlp['ubg'])
    trajectories = ca.Function('trajectories', [nlp['w']], [plt_data['x_plot'], plt_data['z_plot'], plt_data['theta_plot'], plt_data['slack_tether_plot'], plt_data['slack_dae_plot']], ['w'], ['x', 'z', 'theta', 's_tether', 's_dae'])
    x_opt, z_opt, theta_opt, slack_tether, slack_dae = trajectories(sol['x'])
    print('=======================================================================')
    # scale the measurements
    n_grid = x_opt.shape[1]
    x_opt_rescaled = ca.DM.zeros(x_opt.shape)
    z_opt_rescaled = ca.DM.zeros(z_opt.shape)
    for i in range(n_grid):
        x_opt_rescaled[:, i] = get_reverse_rescaled_vars(model, x=x_opt[:, i])
    for i in range(n_grid-1):
        z_opt_rescaled[:, i] = get_reverse_rescaled_vars(model, z=z_opt[:, i])
    x_opt_rescaled = x_opt_rescaled.full() # to numpy array
    z_opt_rescaled = z_opt_rescaled.full() # to numpy array
    

    dt      = t_meas_main[1] - t_meas_main[0]     
    t_grid  = np.linspace(0, dt*(Nm-1), n_grid)

    # Plot the results                         
    constraints_l_t, constraints_dl_t = np.zeros(n_grid), np.zeros(n_grid)
    for i in range(n_grid):
        tether_cons = tether_constraints(model, x_opt[:, i])
        constraints_l_t[i] = tether_cons[0]
        constraints_dl_t[i] = tether_cons[1]
        print(f"tether constraints at time {i}: {tether_cons}")
    print('=======================================================================')
    print(f'p_{1}* = ', theta_opt[0,:])
    print(f'p_{2}* = ',theta_opt[1, :])
    print('=======================================================================')

    fig_q, ax_q = plot_xy_mixed([t_grid[:], t_meas_main[:Nm]], [[x_opt_rescaled[0,:], x_opt_rescaled[1,:], x_opt_rescaled[2,:]], [ x[:Nm], y[:Nm], z[:Nm]]], labels_groups=[['x_opt_rescaled_1', 'x_opt_rescaled_2', 'x_opt_rescaled_3'],['x', 'y', 'z']], xlabel='time (s)', ylabel='position (m) ', title='kite position from collocation and measurment values (filtered)')
    fig_q_3d, ax_q_3d = plot_3d_mixed([[(x_opt_rescaled[0,:], x_opt_rescaled[1,:], x_opt_rescaled[2,:])], [(x[:Nm], y[:Nm], z[:Nm])]], labels_groups=[['q_opt_rescaled'], ['q_mesured' ]],title='kite position from collocation and measurment values (filtered)')
    fig_v, ax_v = plot_xy_mixed([t_grid[:], t_meas_main[:Nm]], [[x_opt_rescaled[3,:], x_opt_rescaled[4,:], x_opt_rescaled[5,:]], [v_x[:Nm], v_y[:Nm], v_z[:Nm]]], labels_groups=[['v_x_opt_rescaled', 'v_y_opt_rescaled', 'v_z_opt_rescaled'], ['v_x', 'v_y', 'v_z']], xlabel='time (s)', ylabel='velocity (m/s) ', title='kite velocity from collocation and measurment values (filtered)')
    fig_u, ax_u = plot_xy_mixed([t_grid[:], t_meas_main[:Nm]], [[x_opt_rescaled[6,:], x_opt_rescaled[7,:]], [u_s[:Nm]/100, u_d[:Nm]/100]], labels_groups=[['u_s_opt_rescaled', 'u_d_opt_rescaled'], ['u_s', 'u_d']], xlabel='time (s)', ylabel='steering ', title='kite steering from collocation and measurment values (filtered)')
    fig_l_t, ax_l_t = plot_xy_mixed([t_grid[:], t_meas_main[:Nm]], [[x_opt_rescaled[8,:]], [l_t_with_offset[:Nm]]], labels_groups=[['l_t_opt_rescaled'], ['l_t']], xlabel='time (s)', ylabel='tether length (m) ', title='kite tether length from collocation and measurment values (filtered)')
    fig_dl_t, ax_dl_t = plot_xy_mixed([t_grid[:], t_meas_main[:Nm]], [[x_opt_rescaled[9,:]], [ dl_t[:Nm]]], labels_groups=[['dl_t_opt_rescaled'], [ 'dl_t']], xlabel='time (s)', ylabel='tether velocity (m/s) ', title='tether reelout velocity')

    fig_c, ax_c = plot_xy(t_grid[:], [constraints_l_t, constraints_dl_t], labels=['c', 'c_dot'], xlabel='time (s)', ylabel='tether constraints', title='tether constraints over time')
    fig_z, ax_z = plot_xy(t_grid[1:], [z_opt_rescaled[0,:]], labels=['z_opt'], xlabel='time (s)', ylabel='z []', title='z over time')
    # fig_tether_slacks, ax_tether_slacks = plot_xy(t_grid[:], [slack_tether[0,:].full().flatten(), slack_tether[1,:].full().flatten()], labels=['s_c0',  's_c1'], xlabel='time (s)', ylabel='tether slack values ', title='tether slacks')
    # fig_dae_slacks, ax_dae_slacks = plot_xy(t_grid[1:], [slack_dae[0,:].full().flatten(),
    #                                                     slack_dae[1,:].full().flatten(),
    #                                                     slack_dae[2,:].full().flatten(),
    #                                                     slack_dae[3,:].full().flatten(),
    #                                                     slack_dae[4,:].full().flatten(),
    #                                                     slack_dae[5,:].full().flatten(),
    #                                                     slack_dae[6,:].full().flatten(),
    #                                                     slack_dae[7,:].full().flatten(),
    #                                                     slack_dae[8,:].full().flatten(),
    #                                                     slack_dae[9,:].full().flatten()],
    #                                                     labels=['s_x',  's_y', 's_z', 's_vx', 's_vy', 's_vz', 's_us', 's_ud', 's_l_t','s_dl_t' ],
    #                                                     xlabel='time (s)', ylabel='dae slack values ', title='dae slacks')
# 


    plt.show()

                  
                


    




