import awebox.mdl.model as mdl
import awebox.mdl.architecture as archi
import awebox.opts.options as opts
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import awebox.opts.kite_data.kitepower_lei_data as kitepower_lei_data
from  measurement_processing import rotate_enu, remove_outliers, interpolate_data, noise_estimation, get_weighted_cov
from awebox.opts.kite_data.kitepower_lei_data import data_dict as data_dict_func
import casadi as ca
import numpy as np
import json, os
from typing import Optional
import settings


# Load the kite geometry parameters dictionary
data_dict = data_dict_func()

def setup_model(measurement_data: dict):
    """
    Setup the kite model based on the provided measurement data (wind velocity in this case).
    """
    upwind_velocity_without_outliers = remove_outliers(
        measurement_data['est_wind_velocity'], 10, 2
    )
    upwind_velocity_filtered = interpolate_data(upwind_velocity_without_outliers)

    # Define the model options for Lei kite model
    options_seed = {}
    options_seed['user_options.wind.u_ref'] = np.mean(upwind_velocity_filtered)
    options_seed = ampyx_ap2_settings.set_kitepower_lei_settings(options_seed)

    options = opts.Options()
    options.fill_in_seed(options_seed)

    # Set the kite architecture
    model      = mdl.Model()
    architecture = archi.Architecture(options['user_options']['system_model']['architecture'])
    options.build(architecture)
    model.build(options['model'], architecture)

    print("----------------------------------------------------------------------------")
    print('Wind model options:')
    print(f"Wind model: {options_seed['user_options.wind.model']}")
    print(f"Wind reference height: {options_seed['params.wind.z_ref']} m")
    print(f"Using wind reference velocity: {options_seed['user_options.wind.u_ref']} m/s") 
    print("----------------------------------------------------------------------------") 

    return model, options

def generate_implicit_dae_F(model, options, np, params_dict):


    model_constraints_list = model.constraints_list
    mdl_vars = model.variables
    mdl_params = model.parameters

    # create functions
    fun_opts = {'construction': {'jit_code_gen': {'include': False}}}
    mdl_eq_fun = model_constraints_list.get_function(fun_opts, mdl_vars, mdl_params, 'eq')

    # create OCP variables
    nx = mdl_vars['x'].shape[0]
    nu = mdl_vars['u'].shape[0]  -3 # remove fictitious forces
    nz = mdl_vars['z'].shape[0]

    xdot = ca.SX.sym('xdot', nx)
    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', nu)
    z = ca.SX.sym('z', nz)
    p = ca.SX.sym('p', np)

    # fill in AWEbox variables and parameters
    theta = model.variables_dict['theta'](1.0)
    theta['diam_t'] = (14 * 1e-3) / model.scaling['theta', 'diam_t'] # TODO: insert correct diameter (scaled)
    

    var_list = []
    for var in list(mdl_vars.keys()):
        if var == 'x':
            var_list.append(x)
        elif var == 'u':
            var_list.append(ca.vertcat(u,u)) # set fictitious forces to zero
        elif var == 'z':
            var_list.append(z)
        elif var == 'xdot':
            var_list.append(xdot)
        elif var == 'theta':
            var_list.append(theta)
    F_vars = ca.vertcat(*var_list)

    # fill in physical parameters / numerical values
    theta0 = model.parameters_dict['theta0'](0.0)
    for k, v in options['model']['params'].items():
        if type(v) == dict:
            for k1, v1 in v.items():
                if type(v1) == dict:
                    for k2, v2 in v1.items():
                        if type(v2) != dict:
                            theta0[k, k1, k2] = v2
                        else:
                            raise ValueError('Recursion depth not deep enough!')
                else:
                    theta0[k, k1] = v1
        else:
            theta0[k] = v

    # fill in symbolic parameters
    theta0_list = []
    counter = 0
    for k in range(theta0.shape[0]):
        CanIdx = theta0.getCanonicalIndex(k)
        # print(CanIdx)
        if len(CanIdx) == 2:
            try:
                test = params_dict[CanIdx[0]][CanIdx[1]]
                theta0_list.append(p[counter])
                counter += 1
                print('Symbolic parameter added!')
            except:
                theta0_list.append(theta0.cat[k])
        elif len(CanIdx) == 3:
            try:
                test = params_dict[CanIdx[0]][CanIdx[1]][CanIdx[2]]
                theta0_list.append(p[counter])
                counter += 1
                print('Symbolic parameter added!')

            except:
                theta0_list.append(theta0.cat[k])
        elif len(CanIdx) == 4:
            try:
                test = params_dict[CanIdx[0]][CanIdx[1]][CanIdx[2]][CanIdx[3]]
                theta0_list.append(p[counter])
                counter += 1
                print('Symbolic parameter added!')

            except:
                theta0_list.append(theta0.cat[k]) 
    if counter != np:
        raise ValueError('Symbolic parameters failed')
    theta0_sym = theta0(ca.vertcat(*theta0_list))

    # create F_params
    param_list = []
    for var_type in list(mdl_params.keys()):
        if var_type == 'phi':
            param_list.append(model.parameters_dict['phi'](0.0))
        if var_type == 'theta0':
            param_list.append(theta0_sym)

    F_params = mdl_params(ca.vertcat(*param_list))

    # evaluate functions to use in OCP
    mdl_eq_expr = mdl_eq_fun(F_vars, F_params)

    # make new function
    F_dae = ca.Function('F_dae', [xdot, x, u, z, p], [mdl_eq_expr])

    return F_dae

def get_bounds():
    """
    Returns the lower and upper bounds for the states, controls, and algebraic variables.
    """
    # Lower-Bounds
    lbs = {
        'x': {
            'q':    ca.DM([-ca.inf, -ca.inf, 10.0]),
            'dq':   ca.DM([ -ca.inf, -ca.inf, -ca.inf ]),
            'u_s':  ca.DM([ -1. ]),
            'u_d':  ca.DM([ 0.0 ]),
            'l_t':  ca.DM([ 1.0e-2]),
            'dl_t': ca.DM([ -30.0 ]),
        },
        'u': {
            'du_s': ca.DM([ -.8 ]),
            'du_d': ca.DM([ -1. ]),
            'ddlt': ca.DM([ -2.0 ]),
        },
        'z': {
            'lambda': ca.DM([ 1]),
        },  
        'p': {
            'K_s,D': ca.DM([ 0.0 ]),
        }
    }

    # Upper-Bounds
    ubs = {
        'x': {
            'q':    ca.DM([ca.inf, ca.inf, ca.inf]),
            'dq':   ca.DM([ ca.inf, ca.inf, ca.inf]),
            'u_s':  ca.DM([ 1. ]),
            'u_d':  ca.DM([ 1.]),
            'l_t':  ca.DM([ 1.0e3 ]),
            'dl_t': ca.DM([ 5.0]),
        },
        'u': {
            'du_s': ca.DM([ 1. ]),
            'du_d': ca.DM([ 0.1 ]),
            'ddlt': ca.DM([ 2. ]),
        },
        'z': {
            'lambda': ca.DM([ ca.inf]),
        },
        'p': {
            'K_s,D': ca.DM([ 2.0 ]),
        }
    }

    return lbs, ubs


def get_scaled_bounds(model):
    """
    Returns the lower and upper bounds for the states, controls, and algebraic variables "scaled".
    """

    # Lower-Bounds
    lbs = {
        'x': {
            'q':    ca.DM([-ca.inf/model.scaling['x'][0], -ca.inf/model.scaling['x'][1], 10.0/model.scaling['x'][2]]),
            'dq':   ca.DM([ -ca.inf/model.scaling['x'][3], -ca.inf/model.scaling['x'][4], -ca.inf/model.scaling['x'][5] ]),
            'u_s':  ca.DM([ -1.0 /model.scaling['x'][6]]),
            'u_d':  ca.DM([ 0.0 /model.scaling['x'][7]]),
            'l_t':  ca.DM([ 1.0e-2/	model.scaling['x'][8]]),
            'dl_t': ca.DM([ -10.0 / model.scaling['x'][9]]),
        },
        'u': {
            'du_s': ca.DM([ -1. / model.scaling['u'][3]]),
            'du_d': ca.DM([ -0.1 / model.scaling['u'][4]]),
            'ddlt': ca.DM([ -2. / model.scaling['u'][5]]),
        },
        'z': {
            'lambda': ca.DM([ 1e-3 /model.scaling['z'][0]]),
        },  
        'p': {
            #'K_s_D': ca.DM([ 0.0 ]),
            'c_s' : ca.DM([-np.deg2rad(60)/0.6]),
        }
    }

    # Upper-Bounds
    ubs = {
        'x': {
            'q':    ca.DM([ca.inf/model.scaling['x'][0], ca.inf/model.scaling['x'][1], ca.inf/model.scaling['x'][2]]),
            'dq':   ca.DM([ ca.inf /model.scaling['x'][3], ca.inf/model.scaling['x'][4], ca.inf/model.scaling['x'][5]]),
            'u_s':  ca.DM([ 1. / model.scaling['x'][6]]),    
            'u_d':  ca.DM([ 1. / model.scaling['x'][7]]),
            'l_t':  ca.DM([ 1.0e3 / model.scaling['x'][8]]), 
            'dl_t': ca.DM([ 10.0 / model.scaling['x'][9]]),
        },
        'u': {
            'du_s': ca.DM([ 0.1 / model.scaling['u'][3]]),
            'du_d': ca.DM([ .1 / model.scaling['u'][4]]),
            'ddlt': ca.DM([ 2. / model.scaling['u'][5]]),
        },
        'z': {
            'lambda': ca.DM([ ca.inf /model.scaling['z'][0]]),
        },
        'p': {
            #'K_s,D': ca.DM([ 1.0 ]),
            'c_s' : ca.DM([np.deg2rad(60)/0.6]),
        }
    }

    return lbs, ubs

def get_scaled_vars(model, x: Optional[ca.DM] = None, u: Optional[ca.DM] = None, z: Optional[ca.DM] = None):
    """
    Scales the variables x, u, and z according to the model scaling.
    """
    
    # If nothing provided, error out
    if x is None and u is None and z is None:
        raise ValueError("At least one of x, u, or z must be provided.")

    outputs = []

    if x is not None:
        x_scaled = x / model.scaling['x']
        outputs.append(x_scaled)

    if u is not None:
        u_scaled = u / model.scaling['u'][3:]
        outputs.append(u_scaled)

    if z is not None:
        z_scaled = z / model.scaling['z']
        outputs.append(z_scaled)

    # Return single ca.DM if only one, otherwise a tuple
    return outputs[0] if len(outputs) == 1 else tuple(outputs)

def get_reverse_rescaled_vars(model, x: Optional[ca.DM] = None, u: Optional[ca.DM] = None, z: Optional[ca.DM] = None):
    """
    Scales the variables x, u, and z according to the model scaling.
    """
    
    # If nothing provided, error out
    if x is None and u is None and z is None:
        raise ValueError("At least one of x, u, or z must be provided.")

    outputs = []

    if x is not None:
        x_scaled = x * model.scaling['x']
        outputs.append(x_scaled)

    if u is not None:
        u_scaled = u * model.scaling['u'][3:]
        outputs.append(u_scaled)

    if z is not None:
        z_scaled = z * model.scaling['z']
        outputs.append(z_scaled)

    # Return single ca.DM if only one, otherwise a tuple
    return outputs[0] if len(outputs) == 1 else tuple(outputs)




def flatten_group_bounds(lbs: dict, ubs: dict, group: str):
    """
    Flattens the lower and upper bounds for a given group of variables.
    """
    lb_vec = ca.vertcat(*[ca.DM(lbs[group][var]) for var in lbs[group]])
    ub_vec = ca.vertcat(*[ca.DM(ubs[group][var]) for var in ubs[group]])
    return lb_vec, ub_vec



if __name__ == '__main__':

    # Define path to measurements dataset
    data = settings.load_measurement_data(settings.MEAS_FILE)
    model, options = setup_model(data)

    n_p = 0
    params_dict = {}
    params_dict['geometry'] = {}
    #params_dict['geometry']['m_k'] = [1]
    #np += 1
    params_dict['geometry']['K_s,D'] = [1]
    n_p += 1

    lb, ub = get_bounds()
    lb_x, ub_x = flatten_group_bounds(lb, ub, 'x')
    lb_u, ub_u = flatten_group_bounds(lb, ub, 'u')
    lb_z, ub_z = flatten_group_bounds(lb, ub, 'z')
    lb_p, ub_p = flatten_group_bounds(lb, ub, 'p')
    print(lb_x, ub_x)
    print(lb_u, ub_u)
    print(lb_z, ub_z)
    print(lb_p, ub_p)

    lb_scaled, ub_scaled = get_scaled_bounds(model)
    lb_x_scaled, ub_x_scaled = flatten_group_bounds(lb_scaled, ub_scaled, 'x')
    lb_u_scaled, ub_u_scaled = flatten_group_bounds(lb_scaled, ub_scaled, 'u')
    lb_z_scaled, ub_z_scaled = flatten_group_bounds(lb_scaled, ub_scaled, 'z')
    lb_p_scaled, ub_p_scaled = flatten_group_bounds(lb_scaled, ub_scaled, 'p')
    print(lb_x_scaled, ub_x_scaled)
    print(lb_u_scaled, ub_u_scaled)
    print(lb_z_scaled, ub_z_scaled)
    

    test = generate_implicit_dae_F(model, options, n_p, params_dict)
    print(test)
    print(params_dict)