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
from settings import default_options




class KiteCollocationRunner:
    """
    Encapsulates model setup, data preprocessing, collocation problem setup and solution, and plotting.
    """
    def __init__(self, json_path, num_stages=1, num_finite_elements=1):
        # Paths and data
        self.json_path = pathlib.Path(json_path)
        self.data_dict   = data_dict_func()
        # Collocation settings
        self.num_stages = num_stages
        self.num_finite_elements = num_finite_elements
        # Placeholders
        self.model = None
        self.opts = None
        self.t = None
        self.y_s = None
        self.u_s = None
        self.s0 = None
        self.a0 = None
        self.W_y = None
        self.W_th = None
        self.theta0 = None
        # Collocation problem
        self.nlp_problem = None
        self.nlp_solver = None
        self.plot_data = None
        # Trajectories
        self.x_opt = None
        self.z_opt = None
        self.theta_opt = None
        self.slack_t = None
        self.slack_d = None

        # Initialization steps
        self._load_dll()
        self._setup_model()

        # load collocation options
        opts = default_options()
        self.coll_opts = opts['collocation']
        self.solver_opts = opts['solver']
        self.plot_opts   = opts['plot'] 
    
    

    def _load_dll(self):
        dll_dir = pathlib.Path(
            r"C:\Users\maher\OneDrive\Desktop\Masterarbeit\Code2.0\awebox_kite_power\toolchain\bin"
        )
        if sys.version_info >= (3, 8):
            os.add_dll_directory(str(dll_dir))
        ctypes.CDLL(str(dll_dir / "libhsl.dll"))
        print("libhsl.dll erfolgreich geladen")

    def tether_constraints(self, x_scaled):

        x_pysical = get_reverse_rescaled_vars(self.model, x=x_scaled)
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


    def _setup_model(self):
        # Wind reference
        # wind_clean = remove_outliers(
        #     self.dict_data['ground_wind_velocity'], 50, 2)
        # wind_interp = interpolate_data(wind_clean)
        upwind_velocity_mean = 6.0#float(np.mean(wind_interp))

        options_seed = {} 
        options_seed['user_options.wind.u_ref'] = upwind_velocity_mean
        options_seed = ampyx_ap2_settings.set_kitepower_lei_settings(options_seed)
        options = opts.Options()
        options.fill_in_seed(options_seed)

       
        model = mdl.Model()
        arch = archi.Architecture(
            options['user_options']['system_model']['architecture']
        )
        options.build(arch)
        model.build(options['model'], arch)

        self.model = model
        self.opts = options

    def load_and_preprocess(self):
        with open(self.json_path, 'r') as f:
            measurement_data = json.load(f)
        # Time vector
        t = np.array(measurement_data['time']) - measurement_data['time'][0]
        # Direction
        # states
        upwind_direction_without_outliers = remove_outliers(measurement_data['ground_upwind_direction'], 50, 100)
        upwind_direction_filtered = interpolate_data(upwind_direction_without_outliers)
        upwind_direction_mean = np.mean(upwind_direction_filtered)
        upwind_direction_mean_vec = np.full(len(t), upwind_direction_mean)
        upwind_velocity_without_outliers = remove_outliers(measurement_data['ground_wind_velocity'], 50, 2)
        upwind_velocity_filtered = interpolate_data(upwind_velocity_without_outliers)
        # Positions & velocities
        pos_x, pos_y, pos_z = np.array(
            [rotate_enu(a, e, n, u) for a, e, n, u in zip(
                upwind_direction_mean_vec,
                measurement_data['kite_pos_east'],
                measurement_data['kite_pos_north'],
                measurement_data['kite_height']
            )]).T

        vel_x, vel_y, vel_z = np.array(
            [rotate_enu(a, e, n, u) for a, e, n, u in zip(
                upwind_direction_mean_vec,
                measurement_data['kite_est_vx'],
                measurement_data['kite_est_vy'],
                measurement_data['kite_est_vz']
            )]).T
        
        # Controls
        steering = np.array(measurement_data['kite_actual_steering'])/100 
        depower = np.array(measurement_data['kite_actual_depower']) /100

        # Tether length + Kalman offset

        tether_length = (
            np.array(measurement_data['ground_tether_length']) +
            self.data_dict['geometry']['h_bridle'] +
            self.data_dict['geometry']['h_kite']
        )
        kf_len, kf_vel = kalman_filter_derivation(t, tether_length)
        offset = float(np.mean(measurement_data['kite_distance']) - np.mean(kf_len))
        tether_length += offset
        reel_vel = np.array(measurement_data['ground_tether_reelout_speed'])
        # Measurement vectors
        y_meas = ca.DM([
            pos_x, pos_y, pos_z,
            vel_x, vel_y, vel_z,
            steering, depower,
            tether_length, reel_vel
        ])

        kf_steering, kf_steering_deriv = kalman_filter_derivation(t, measurement_data['kite_actual_steering'])
        kf_depower, kf_depower_deriv = kalman_filter_derivation(t, measurement_data['kite_actual_depower'])
        noise = noise_estimation(
            t,
            measurement_data['ground_tether_length'],
            measurement_data['ground_tether_reelout_speed']
        )
        KF_results = kalman_filter_for_tether(t,
                                              measurement_data['ground_tether_length'],
                                              measurement_data['ground_tether_reelout_speed'],
                                                noise)
        reel_acc = KF_results['estimated_acceleration']

        # using mesurement claculated controls
        u_meas = ca.DM([kf_steering_deriv, kf_depower_deriv, reel_acc])

        # Scaling
        y_scaled = ca.DM.zeros(y_meas.shape)
        u_scaled = ca.DM.zeros(u_meas.shape)
        for i in range(y_meas.shape[0]):
            y_scaled[:, i] = get_scaled_vars(self.model, x=y_meas[:,i])
        for i in range(u_meas.shape[0]):
            u_scaled[:, i] = get_scaled_vars(self.model, u=u_meas[:,i])
        # Initial conditions
        state0 = y_meas[:,0]
        alg0 = ca.DM([1.0])
        s0, a0 = get_scaled_vars(self.model, x=state0, z=alg0)
        # Weights & theta
        W_y, _ = get_weighted_cov(y_meas, window_length=21, polyorder=3)
        W_th = np.diag([1e-3])
        theta0 = ca.DM([1.0])
        # Store
        self.t = t
        self.y_s = y_scaled
        self.u_s = u_scaled
        self.s0 = s0
        self.a0 = a0
        self.W_y = W_y
        self.W_th = W_th
        self.theta0 = theta0

    def setup_collocation(self):
        """
        Builds the collocation NLP problem dictionary and bounds.
        """
        # define the bucher tableau coefficients:
        B, C, D, _ = generate_butcher_tableau_integral(self.num_stages, "radau")
        
        N = self.coll_opts['N'] -1

        
        # Time step of the measurement
        dt = self.t[1] - self.t[0]

        # Time horizon
        T = dt * (N)

        # discretization of the time horizon
        h = dt / self.num_finite_elements

        # define the scaled bounds
        lb_s, ub_s = get_scaled_bounds(self.model)
        lb_x, ub_x = flatten_group_bounds(lb_s, ub_s, 'x')
        lb_z, ub_z = flatten_group_bounds(lb_s, ub_s, 'z')
        lb_p, ub_p = flatten_group_bounds(lb_s, ub_s, 'p')

        # implicit DAE
        n_param = 0
        params_dict = {}
        params_dict['geometry'] = {}
        # params_dict['geometry']['K_s_D'] = [1]
        # n_param += 1
        params_dict['geometry']['c_s'] = [1]
        n_param += 1
        F_dae = generate_implicit_dae_F(n_param, params_dict)

        # Start with an empty NLP
        w, w0_list, lbw_list, ubw_list = [], [], [], []
        g, lbg_list, ubg_list = [], [], []
        obj = 0
        # for plotting x, z and theta
        x_plot_list, z_plot_list, theta_plot_list = [], [], []
        # for plotting  the slack veriables
        slack_t_plot, slack_d_plot = [], []
        

        # initial
        Xk = ca.SX.sym('X0', self.y_s.shape[0])
        
        w.append(Xk) 
        lbw_list.append(lb_x[0:6])
        lbw_list.append(self.s0[6:8])
        lbw_list.append(lb_x[8:10])
        ubw_list.append(ub_x[0:6])
        ubw_list.append(self.s0[6:8])
        ubw_list.append(ub_x[8:10])

        w0_list.append(self.s0)
        x_plot_list.append(Xk)
    
        # Enforce tether constraints at start
        g.append(self.tether_constraints(Xk)); lbg_list.append(ca.DM.zeros(2)); ubg_list.append(ca.DM.zeros(2))

        #define the initial conditions for the parameters
        theta = ca.SX.sym('theta', self.theta0.shape[0])
        w.append(theta)
        lbw_list.append(lb_p)
        ubw_list.append(ub_p)
        w0_list.append(self.theta0)
        theta_plot_list.append(theta)

        for k in range(N):
            # Loop over integration steps / finite elements
            for i_fe in range(self.num_finite_elements):
                Xc, Zc = [], []
                # State at collocation points
                for j in range(self.num_stages):
                    Xkj = ca.SX.sym(f'X_{k}_{i_fe}_{j}', self.y_s.shape[0])
                    Zkj = ca.SX.sym(f'Z_{k}_{i_fe}_{j}', 1)
                    w += [Xkj, Zkj]
                    lbw_list += [lb_x, lb_z]
                    ubw_list += [ub_x, ub_z]
                    w0_list += [self.y_s[:, k], self.a0]
                    Xc.append(Xkj)
                    Zc.append(Zkj)
                Xk_end = D[0]*Xk
                Zk_end = Zc[-1] 

                # Loop over collocation points
                for j in range(1, self.num_stages+1):
                    xp = C[0,j]*Xk + sum(C[r+1,j]*Xc[r] for r in range(self.num_stages))
                    f_val = F_dae(xp/h, Xc[j-1], self.u_s[:,k], Zc[j-1], theta)
                    g.append(f_val)
                    lbg_list.append(ca.DM.zeros(self.y_s.shape[0]+1))
                    ubg_list.append(ca.DM.zeros(self.y_s.shape[0]+1))
                    Xk_end += D[j]*Xc[j-1]
                Xk = ca.SX.sym(f'X_{k+1}', self.y_s.shape[0])
                Zk = ca.SX.sym(f'Z_{k+1}', 1)
                w += [Xk, Zk]; lbw_list += [lb_x, lb_z]; ubw_list += [ub_x, ub_z]; w0_list += [self.s0, self.a0]
                x_plot_list += [Xk]; z_plot_list += [Zk]

                g.append(Xk - Xk_end)
                lbg_list.append(ca.DM.zeros(self.y_s.shape[0]))
                ubg_list.append(ca.DM.zeros(self.y_s.shape[0]))

                g.append(Zk - Zk_end)
                lbg_list.append(ca.DM.zeros(1))
                ubg_list.append(ca.DM.zeros(1))
            obj += ((self.y_s[:,k+1] - Xk_end).T @ np.eye(10) @ (self.y_s[:,k+1] - Xk_end))
            #obj += (theta - self.theta0).T @ self.W_th @ (theta - self.theta0)

        w = ca.vertcat(*w)
        g = ca.vertcat(*[ca.reshape(gi, gi.numel(), 1) for gi in g])
        w0 = ca.vertcat(*w0_list)
        lbw = np.concatenate(lbw_list)
        ubw = np.concatenate(ubw_list)
        lbg = ca.vertcat(*lbg_list)
        ubg = ca.vertcat(*ubg_list)

        x_plot = ca.horzcat(*x_plot_list)
        z_plot = ca.horzcat(*z_plot_list)
        theta_plot = ca.horzcat(*theta_plot_list)
        slack_t_plot = ca.horzcat(*slack_t_plot)
        slack_d_plot = ca.horzcat(*slack_d_plot)

        self.nlp_dict = {'x': w, 'f': obj, 'g': g}
        self.w0 = w0
        self.lbw = lbw
        self.ubw = ubw
        self.lbg = lbg
        self.ubg = ubg
        self.plot_data = {
            'x_plot': x_plot,
            'z_plot': z_plot,
            'theta_plot': theta_plot,
            'slack_tether_plot': slack_t_plot,
            'slack_dae_plot': slack_d_plot
        }

    def run_collocation(self):
        """
        Solves the previously set up NLP and extracts optimal trajectories.
        """
        if self.nlp_solver is None:
            ipopt_opts = dict(self.solver_opts)
            self.nlp_solver = ca.nlpsol('solver', 'ipopt', self.nlp_dict, ipopt_opts)
        sol = self.nlp_solver(
            x0=self.w0,
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg,
            ubg=self.ubg
        )
        traj = ca.Function(
            'traj', [self.nlp_dict['x']],
            [
                self.plot_data['x_plot'],
                self.plot_data['z_plot'],
                self.plot_data['theta_plot'],
                self.plot_data['slack_tether_plot'],
                self.plot_data['slack_dae_plot']
            ]
        )
        self.x_opt, self.z_opt, self.theta_opt, self.slack_t, self.slack_d = traj(sol['x'])

    def plot_results(self):
        """
        Generate and display all relevant plots comparing collocation results and measurements.
        """
        # Number of optimization grid points
        n_grid = self.x_opt.shape[1]
        N = self.coll_opts['N']
        # Preallocate and reverse-scale optimal states
        x_opt_rescaled = np.zeros((self.y_s.shape[0], n_grid))
        for i in range(n_grid):
            x_opt_rescaled[:, i] = get_reverse_rescaled_vars(
                self.model, x=self.x_opt[:, i]
            ).full().flatten()

        # Time grids
        dt = float(self.t[1] - self.t[0])
        t_opt = np.linspace(0, dt * (n_grid - 1), n_grid)
        n_meas = N 
        t_meas = self.t[:n_meas]

        # Reverse-scale measurements (positions)
        q_meas = np.zeros((3, n_meas))
        for i in range(n_meas):
            q_meas[:, i] = get_reverse_rescaled_vars(
                self.model, x=self.y_s[:, i]
            ).full().flatten()[:3]

        # Plot positions: collocation vs. measurement
        plot_xy_mixed(
            [t_opt, t_meas],
            [x_opt_rescaled[:3, :], q_meas],
            labels_groups=[['x_opt','y_opt','z_opt'], ['x_meas','y_meas','z_meas']],
            xlabel='time (s)', ylabel='position (m)',
            title='Kite Position: Collocation vs. Measurement'
        )

        # 3D trajectory comparison
        plot_3d_mixed(
            [
                [(x_opt_rescaled[0,:], x_opt_rescaled[1,:], x_opt_rescaled[2,:])],
                [(q_meas[0,:], q_meas[1,:], q_meas[2,:])]
            ],
            [
                ['q_opt'],
                ['q_meas']
            ],
            title='3D Kite Position Comparison'
        )
        

        # Plot velocities: collocation vs measurement
        v_meas = np.zeros((3, n_meas))
        for i in range(n_meas):
            full_x = get_reverse_rescaled_vars(
                self.model, x=self.y_s[:, i]
            ).full().flatten()
            v_meas[:, i] = full_x[3:6]
        plot_xy_mixed(
            [t_opt, t_meas],
            [x_opt_rescaled[3:6, :], v_meas],
            labels_groups=[['vx_opt','vy_opt','vz_opt'], ['vx_meas','vy_meas','vz_meas']],
            xlabel='time (s)', ylabel='velocity (m/s)',
            title='Kite Velocity: Collocation vs. Measurement'
        )

        # Plot control inputs: steering & depower
        u_meas = np.zeros((2, n_meas))
        for i in range(n_meas):
            full_x = get_reverse_rescaled_vars(
                self.model, x=self.y_s[:, i]
            ).full().flatten()
            u_meas[:, i] = full_x[6:8]
        plot_xy_mixed(
            [t_opt, t_meas],
            [x_opt_rescaled[6:8, :], u_meas],
            labels_groups=[['steering_opt','depower_opt'], ['steering_meas','depower_meas']],
            xlabel='time (s)', ylabel='control input',
            title='Control Inputs: Collocation vs. Measurement'
        )

        # Plot tether length l_t
        # index 8 of state vector is tether length
        l_t_opt = x_opt_rescaled[8:9, :]
        l_t_meas = np.zeros(n_meas)
        for i in range(n_meas):
            l_t_meas[i] = get_reverse_rescaled_vars(self.model, x=self.y_s[:, i]).full().flatten()[8]
        plot_xy_mixed(
            [t_opt, t_meas],
            [l_t_opt, l_t_meas.reshape(1, -1)],
            labels_groups=[['l_t_opt'], ['l_t_meas']],
            xlabel='time (s)', ylabel='tether length (m)',
            title='Tether Length: Collocation vs. Measurement'
        )

        # Plot tether velocity dl_t
        # index 9 of state vector is tether speed
        dl_t_opt = x_opt_rescaled[9:10, :]
        dl_t_meas = np.zeros(n_meas)
        for i in range(n_meas):
            dl_t_meas[i] = get_reverse_rescaled_vars(self.model, x=self.y_s[:, i]).full().flatten()[9]
        plot_xy_mixed(
            [t_opt, t_meas],
            [dl_t_opt, dl_t_meas.reshape(1, -1)],
            labels_groups=[['dl_t_opt'], ['dl_t_meas']],
            xlabel='time (s)', ylabel='tether reelout velocity (m/s)',
            title='Tether reelout velocity: Collocation vs. Measurement'
        )

        # Plot constraint c 
        constraints_l_t, constraints_dl_t = np.zeros(n_grid), np.zeros(n_grid)
        for i in range(n_grid):
            tether_cons = self.tether_constraints(self.x_opt[:, i])
            constraints_l_t[i] = tether_cons[0]
            constraints_dl_t[i] = tether_cons[1]
        plot_xy(t_opt, 
                [constraints_l_t, constraints_dl_t],
                labels=['c', 'c_dot'], xlabel='time (s)',
                ylabel='tether constraints',
                title='tether constraints over time')


        # Plot algebraic variable z (if present)
        if hasattr(self, 'z_opt') and self.z_opt is not None:
            # reverse-scale algebraic trajectories
            nz = self.z_opt.shape[0]
            z_opt_rescaled = np.zeros((nz, n_grid))
            for i in range(n_grid-1):
                z_opt_rescaled[:, i] = get_reverse_rescaled_vars(
                    self.model, z=self.z_opt[i]
                ).full().flatten()
            plot_xy_mixed(
                [t_opt],
                [z_opt_rescaled],
                labels_groups=[['z_opt']],
                xlabel='time (s)', ylabel='algebraic var',
                title='Algebraic Variable z'
            )

            print('=======================================================================')
            print(f'p_{1}* = ', self.theta_opt[0,:])
            #print(f'p_{2}* = ',self.theta_opt[1, :])
            print('=======================================================================')

        # Optional: plot slack tether variable
        # if hasattr(self, 'slack_t') and self.slack_t is not None:
        #     slack = self.slack_t.full().flatten()
        #     plot_xy_mixed(
        #         [t_opt],
        #         [slack],
        #         labels_groups=[['slack_tether']],
        #         xlabel='time (s)', ylabel='slack',
        #         title='Tether Slack Variable'
        #     )

        # Show or save based on plot options
        if hasattr(self, 'plot_opts') and self.plot_opts.get('save', False):
            plt.savefig(
                f"collocation_results.{self.plot_opts['format']}",
                dpi=self.plot_opts.get('dpi', 150)
            )
        if not hasattr(self, 'plot_opts') or self.plot_opts.get('show', True):
            plt.show()
        else:
            plt.close('all')


    def execute(self):
        """Full pipeline: preprocessing, setup, solve, plot."""
        self.load_and_preprocess()
        self.setup_collocation()
        self.run_collocation()
        self.plot_results()

if __name__ == "__main__":
    runner = KiteCollocationRunner(
        json_path=pathlib.Path(__file__).parent / '..' / '..' / 'Data' / 'DataShots' / 'one_loop_meas_2025_1.json'
    )
    runner.execute()
