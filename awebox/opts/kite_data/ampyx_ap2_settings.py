#!usr/bin/python3

import awebox.tools.print_operations as print_op
import awebox.opts.kite_data.kitepower_lei_data as kitepower_lei_data
import awebox as awe
import numpy as np
import casadi as ca

def set_ampyx_ap2_settings(options):

    # 6DOF Ampyx Ap2 model
    options['user_options.system_model.kite_dof'] = 6
    options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options.trajectory.system_type'] = 'lift_mode'
    options['user_options.trajectory.lift_mode.windings'] = 1

    # tether parameters
    options['params.tether.cd'] = 1.2
    options['params.tether.rho'] = 0.0046*4/(np.pi*0.002**2)
    options['user_options.trajectory.fixed_params'] = {'diam_t': 2e-3}
    options['model.tether.control_var'] = 'ddl_t'  # tether acceleration control

    # tether drag model (more accurate than the Argatov model in Licitra2019)
    options['user_options.tether_drag_model'] = 'multi'
    options['model.tether.aero_elements'] = 5

    # tether force limit
    options['model.model_bounds.tether_stress.include'] = False
    options['model.model_bounds.tether_force.include'] = True
    options['params.model_bounds.tether_force_limits'] = np.array([50, 1800.0])

    # flight envelope
    options['model.model_bounds.airspeed.include'] = True
    options['params.model_bounds.airspeed_limits'] = np.array([10, 32.0])
    options['model.model_bounds.aero_validity.include'] = True
    options['user_options.kite_standard.aero_validity.beta_max_deg'] = 20.
    options['user_options.kite_standard.aero_validity.beta_min_deg'] = -20.
    options['user_options.kite_standard.aero_validity.alpha_max_deg'] = 9.0
    options['user_options.kite_standard.aero_validity.alpha_min_deg'] = -6.0

    # acceleration constraint
    options['model.model_bounds.acceleration.include'] = False

    # aircraft-tether anti-collision
    options['model.model_bounds.rotation.include'] = True
    options['model.model_bounds.rotation.type'] = 'yaw'
    options['params.model_bounds.rot_angles'] = np.array([80.0*np.pi/180., 80.0*np.pi/180., 40.0*np.pi/180.0])

    # variable bounds
    options['model.system_bounds.x.l_t'] = [10.0, 700.0]  # [m]
    options['model.system_bounds.x.dl_t'] = [-15.0, 20.0]  # [m/s]

    options['model.system_bounds.x.ddl_t'] = [-2.4, 2.4]  # [m/s^2]
    options['model.system_bounds.x.q'] = [np.array([-ca.inf, -ca.inf, 100.0]), np.array([ca.inf, ca.inf, ca.inf])]
    options['model.system_bounds.theta.t_f'] = [20., 70.]  # [s]
    options['model.system_bounds.z.lambda'] = [0., ca.inf]  # [N/m]
    omega_bound = 50.0*np.pi/180.0
    options['model.system_bounds.x.omega'] = [np.array(3*[-omega_bound]), np.array(3*[omega_bound])]
    options['user_options.kite_standard.geometry.delta_max'] = np.array([20., 30., 30.]) * np.pi / 180.
    options['user_options.kite_standard.geometry.ddelta_max'] = np.array([2., 2., 2.])

    # don't include induction effects
    options['user_options.induction_model'] = 'not_in_use'

    # initialization
    options['solver.initialization.groundspeed'] = 15.
    options['solver.initialization.inclination_deg'] = 45.
    options['solver.initialization.cone_deg'] = 15.
    options['solver.initialization.l_t'] = 200.



    return options

def set_kitepower_lei_settings(options):
    # indicate desired system architecture
    options['user_options.system_model.architecture'] = {1: 0}
    options['user_options.kite_standard'] = kitepower_lei_data.data_dict()
    options['user_options.system_model.wing_type'] = 'LEI'
    options['user_options.system_model.kite_dof'] = 3
    options['model.tether.control_var'] = 'ddl_t'
    # tether drag model (more accurate than the Argatov model in Licitra2019)
    options['user_options.tether_drag_model'] = 'multi'
    options['model.tether.aero_elements'] = 5

    # tether force limit
    options['model.model_bounds.tether_stress.include'] = True
    options['model.model_bounds.tether_force.include'] = True
    # options['params.model_bounds.tether_force_limits'] = np.array([50, 1800.0])

    # flight envelope
    options['model.model_bounds.airspeed.include'] = True
    # options['params.model_bounds.airspeed_limits'] = np.array([10, 32.0])
    options['model.model_bounds.aero_validity.include'] = False
    options['user_options.kite_standard.aero_validity.beta_max_deg'] = 35.
    options['user_options.kite_standard.aero_validity.beta_min_deg'] = -20.
    options['user_options.kite_standard.aero_validity.alpha_max_deg'] = 20.0
    options['user_options.kite_standard.aero_validity.alpha_min_deg'] = -20.0

    # variable bounds
    options['model.model_bounds.acceleration.include']  = False
    options['model.system_bounds.x.l_t'] = [10.0, 700.0]  # [m]
    options['model.system_bounds.x.dl_t'] = [-15.0, 20.0]  # [m/s]
    options['model.system_bounds.x.ddl_t'] = [-2.4, 2.4]  # [m/s^2]
    #options['model.system_bounds.x.q'] = [np.array([-ca.inf, -ca.inf, 100.0]), np.array([ca.inf, ca.inf, ca.inf])]
    options['model.system_bounds.theta.t_f'] = [20., 70.]  # [s]
    options['model.system_bounds.z.lambda'] = [0., ca.inf]  # [N/m]

    # coefficient boundaries
    #options['model.system_bounds.x.coeff'] =  [np.array([-1., 0.]), np.array([1., 1.])]
    #options['model.system_bounds.u.dcoeff'] =  [np.array([-.8, -1]), np.array([.8, 1])]


    # indicate desired environment
    options['params.wind.z_ref'] = 6.0
    options['params.wind.log_wind.z0_air'] = 0.0002
    options['params.wind.power_wind.exp_ref'] = 0.15
    options['user_options.wind.model'] = 'log_wind'
    #options['user_options.wind.u_ref'] = 6.
    #options['nlp.collocation.u_param'] = 'ploy'



    return options

