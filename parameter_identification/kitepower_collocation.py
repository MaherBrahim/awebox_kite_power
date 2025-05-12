import casadi as ca
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
from plotting import plot_xy, plot_xyz, animate_3d_flight, is_gaussian_noise
from kalman_filter import  kalman_filter_for_tether, kalman_filter_derivation
from  measurement_processing import rotate_enu, remove_outliers, interpolate_data, noise_estimation, get_weighted_cov
from awebox.opts.kite_data.kitepower_lei_data import data_dict as data_dict_func
import awebox.mdl.model as mdl
import awebox.mdl.architecture as archi
import awebox.opts.options as opts
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import os
import json

def setup_model():
    """
    Set up the model and options for the kitepower system.
    """
    # Load the options
    options_seed = {} 
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
json_file = os.path.join(data_path,  "one_loop_meas_2025_3.json")

with open(json_file, "r") as f:
    data = json.load(f)


# define the bounds 
lb, ub = get_bounds()
lb_x, ub_x = flatten_group_bounds(lb, ub, 'x')
lb_u, ub_u = flatten_group_bounds(lb, ub, 'u')
lb_z, ub_z = flatten_group_bounds(lb, ub, 'z')
lb_p, ub_p = flatten_group_bounds(lb, ub, 'p')

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

def setup_collocation(n_s:int, N_fe, y_meas: np.ndarray, 
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


    

    # Continuous time dynamics
    n_param = 0
    params_dict = {}
    params_dict['geometry'] = {}
    params_dict['geometry']['K_s,D'] = [1]
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

    # for plotting x an z 
    x_plot = []
    z_plot = []

    # define the initial conditions for the states
    Xk = ca.SX.sym('X0', nx)
    w.append(Xk)
    lbw.append(lb_x)
    ubw.append(ub_x)
    w0.append(X0)
    x_plot.append(Xk)
    # define the initial conditions for the algebraic variables
    Zk = ca.SX.sym('Z0', nz)
    w.append(Zk)
    lbw.append(lb_z)
    ubw.append(ub_z)
    w0.append(Z0)
    z_plot.append(Zk)

    # Enforce tether constraints at start
    g.append(tether_constraints(model, Xk))
    lbg.append(ca.DM.zeros(2))
    ubg.append(ca.DM.zeros(2))

    #define the initial conditions for the parameters
    theta = ca.SX.sym('theta', n_param)
    w0.append(theta_hat) 
    w.append(theta)
    lbw.append(lb_p)
    ubw.append(ub_p)
    
    



    for k in range(0,N):
        # Loop over integration steps / finite elements
        for i_fe in range(N_fe):
            Xk_end = D[0] * Xk
            Zk_end = D[0] * Zk
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
                w0.append(X0)  
                w0.append(Z0)
            
            

            # Loop over collocation points
            for j in range(1, n_s + 1):
                # Expression for the state derivative at the collocation point
                xp = C[0, j] * Xk
                zp = C[0, j] * Zk
                for r in range(n_s):
                    xp = xp + C[r + 1, j] * Xc[r]
                    zp = zp + C[r + 1, j] * Zc[r]

                # Model equations
                f_dae = F_dae(xp/h, Xc[j - 1], u_meas[:, k], Zc[j - 1], theta)

                # Collocation equations
                g.append(f_dae)
                lbg.append(np.zeros((nx + nz,)))
                ubg.append(np.zeros((nx + nz,)))
                # Add contribution to the end state 
                Xk_end = Xk_end + D[j] * Xc[j - 1]
                Zk_end = Zk_end + D[j] * Zc[j - 1]
                

            objective +=  (((y_meas[:, k+1] - Xk_end).T @ W_y @ (y_meas[:, k+1] - Xk_end)) )  + (theta-theta_hat).T @ W_theta @ (theta-theta_hat)

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

            # # Tether constraints at knot
            # g.append(tether_constraints(model, Xk))
            # lbg.append(ca.DM.zeros(2))
            # ubg.append(ca.DM.zeros(2))
        
    
    
    g_col = [ca.reshape(Gi, Gi.numel(), 1) for Gi in g]
    print(len(g_col))
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
    }
    casadi_nlp = {'f': objective, 'x': w, 'g': g}
    return nlp, casadi_nlp, plt_data


def collocation_for_LSP (n_s, N_fe, y_meas, u_meas,x0, z0, W_y, W_theta,theta_hat):

    
    # setup the collocation problem
    nlp, casadi_nlp, plt_data  = setup_collocation(n_s, N_fe, y_meas, u_meas, x0, z0, W_y, W_theta, theta_hat)

    opts = {"ipopt": {
            "print_level": 5,
            "check_derivatives_for_naninf": "yes"}} 
    solver = ca.nlpsol('solver', 'ipopt', casadi_nlp, opts)


    return nlp, solver, plt_data


if __name__ == "__main__":
    # Load the model and options
    model, options = setup_model()
    # Define the states und inputs from the measurements
    # time t
    t_meas = np.array(data['time']) - data['time'][0]  
    # states
    upwind_direction_without_outliers = remove_outliers(data['ground_upwind_direction'], 50, 100)
    upwind_direction_filtered = interpolate_data(upwind_direction_without_outliers)
    upwind_direction_mean = np.mean(upwind_direction_filtered)
    upwind_direction_mean_vec = np.full(len(t_meas), upwind_direction_mean)
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

    u_s = np.array(data['kite_actual_steering'])/100 
    u_d = np.array(data['kite_actual_depower'])/100
    l_t = np.array(data['ground_tether_length']) 
    dl_t = np.array(data['ground_tether_reelout_speed'])



    l_t_with_offset = l_t + data_dict['geometry']['h_bridle'] + data_dict['geometry']['h_kite']
    kite_distance =  data['kite_distance']
    kite_distance_kf, _ = kalman_filter_derivation(t_meas, l_t_with_offset)
    offset = np.mean(kite_distance) - np.mean(kite_distance_kf)
    l_t_with_offset += offset


    y_meas = ca.DM([x, y, z, v_x, v_y, v_z, u_s, u_d, l_t_with_offset, dl_t])
    


    # inputs
    u_s_kf, du_s_kf  = kalman_filter_derivation(t_meas, u_s)
    u_d_kf, du_d_kf = kalman_filter_derivation(t_meas, u_d)
    noises = noise_estimation(t_meas, l_t, dl_t)
    KF_results = kalman_filter_for_tether(t_meas, l_t, dl_t, noises)
    ddl_t = KF_results['estimated_acceleration']
    
    u_meas = ca.DM([du_s_kf, du_d_kf, ddl_t]) 

    
    # scale the measurements and the inputs
    y_meas_scaled = ca.DM.zeros(y_meas.shape)
    u_meas_scaled = ca.DM.zeros(u_meas.shape)
    for i in range(y_meas.shape[0]):
        y_meas_scaled[:, i] = get_scaled_vars(model, x=y_meas[:, i])
    for i in range(u_meas.shape[0]):
        u_meas_scaled[:, i] = get_scaled_vars(model, u=u_meas[:, i])

    print('y_meas_scaled:', y_meas_scaled)
    print('u_meas_scaled:', u_meas_scaled)

    
    # define the initial states:
    X0 = y_meas[:, 0]
    Z0 = ca.DM([1.0])

    X0_scaled, Z0_scaled = get_scaled_vars(model, x=X0, z=Z0)
    print('X0_scaled:', X0_scaled)
    print('Z0_scaled:', Z0_scaled)

    W_y, weighting_mat = get_weighted_cov(y_meas, window_length=21, polyorder=3)
    W_theta = 1
    theta_hat = ca.DM([0.2])
    
    
    Nm = 5
    N= Nm-1
    # define the number of collocation points and the number of finite elements
    n_s= 3 
    N_fe= 2



    # nlp, casadi_nlp = setup_collocation(N, n_s, N_fe, y_meas, u_meas, x0, z0, W_y, W_theta, theta_hat)

    
    # Call the collocation function
    nlp, solver, plt_data = collocation_for_LSP(n_s, N_fe, y_meas_scaled[:,:Nm], u_meas_scaled[:,:Nm], X0_scaled, Z0_scaled, W_y, W_theta, theta_hat)
    #print(nlp['w0'])
    # Solve the collocation problem
    sol = solver(x0=nlp['w0'],
                 lbx=nlp['lbw'],
                 ubx=nlp['ubw'],
                 lbg=nlp['lbg'],
                 ubg=nlp['ubg'])
    w_opt = sol['x'].full()
    trajectories = ca.Function('trajectories', [nlp['w']], [plt_data['x_plot'], plt_data['z_plot']], ['w'], ['x', 'z'])
    x_opt, z_opt = trajectories(sol['x'])
    print('=======================================================================')
    print('x_opt:', get_reverse_rescaled_vars(model, x=x_opt[:,1]))
    print('=======================================================================')
    # scale the measurements
    n_grid = x_opt.shape[1]
    x_opt_rescaled = ca.DM.zeros(x_opt.shape)
    z_opt_rescaled = ca.DM.zeros(z_opt.shape)
    for i in range(n_grid):
        x_opt_rescaled[:, i] = get_reverse_rescaled_vars(model, x=x_opt[:, i])
    for i in range(n_grid):
        z_opt_rescaled[:, i] = get_reverse_rescaled_vars(model, z=z_opt[:, i])
    x_opt_rescaled = x_opt_rescaled.full() # to numpy array
    z_opt_rescaled = z_opt_rescaled.full() # to numpy array
    

    dt      = t_meas[1] - t_meas[0]     # Δt zwischen Messpunkten
    t_grid  = np.linspace(0, dt*(Nm-1), n_grid)
    # Plot the results                         
    fig_opt, ax_opt = plot_xy(t_grid[:], [x_opt_rescaled[0,:], x_opt_rescaled[1,:], x_opt_rescaled[2,:]], labels=['x_opt_rescaled_1', 'x_opt_rescaled_2', 'x_opt_rescaled_3'], xlabel='time (s)', ylabel='x (m)', title='x over time')
    fig_2d_q, ax_2d_q = plot_xy(t_meas[:Nm], [x[:Nm], y[:Nm], z[:Nm]], labels=['x', 'y', 'z'], xlabel='time (s)', ylabel='position ', title='kite position from measurment values (filtered)')
    fig_z, ax_z = plot_xy(t_grid[:], [z_opt_rescaled[0,:]], labels=['z_opt'], xlabel='time (s)', ylabel='z []', title='z over time')
    print('p* = ',w_opt[-1, :])
    print('=======================================================================')

    for i in range(n_grid):
        tether_cons = tether_constraints(model, x_opt[:, i])
        print(f"tether constraints at time {i}: {tether_cons}")
    print('=======================================================================')


    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(t_grid[:], x_opt_rescaled[8,:], label='optimal tether length')
    plt.plot(t_meas[:Nm], l_t_with_offset[:Nm], label='measured tether length')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

                  
                


    




