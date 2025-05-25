#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2020 Thilo Bronnenmeyer, Kiteswarms Ltd.
#    Copyright (C) 2016      Elena Malz, Sebastien Gros, Chalmers UT.
#
#    awebox is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 3 of the License, or (at your option) any later version.
#
#    awebox is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with awebox; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
'''
specific aerodynamics for a 3dof kite with roll_control
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: jochem de schutter, rachel leuthold, alu-fr 2017-20
'''

import casadi.tools as cas

import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op

import awebox.mdl.aero.kite_dir.frames as frames
import awebox.mdl.aero.kite_dir.tools as tools
import awebox.mdl.aero.indicators as indicators
import numpy as np


from awebox.logger.logger import Logger as awelogger


def get_force_vector(options, variables, atmos, wind, architecture, parameters, kite, outputs):
    kite_dcm = get_kite_dcm(options, variables, wind, kite, architecture, parameters)

    vec_u = tools.get_local_air_velocity_in_earth_frame(options, variables, wind, kite, kite_dcm, architecture,
                                                        parameters, outputs)

    force_found_frame = 'earth'
    force_found_vector = get_force_from_u_sym_in_earth_frame(vec_u, options, variables, kite, atmos, wind, architecture,
                                                             parameters)

    return force_found_vector, force_found_frame, vec_u, kite_dcm


def get_force_cstr(options, variables, atmos, wind, architecture, parameters, outputs):

    cstr_list = cstr_op.MdlConstraintList()

    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]

        force_found_vector, force_found_frame, vec_u, kite_dcm = get_force_vector(options, variables, atmos, wind, architecture, parameters, kite, outputs)

        forces_dict = tools.get_framed_forces(vec_u, kite_dcm, variables, kite, architecture)
        force_framed_variable = forces_dict[force_found_frame]

        f_scale = tools.get_f_scale(parameters, options)

        resi_f_kite = (force_framed_variable - force_found_vector) / f_scale

        f_kite_cstr = cstr_op.Constraint(expr=resi_f_kite,
                                        name='f_aero' + str(kite) + str(parent),
                                        cstr_type='eq')
        cstr_list.append(f_kite_cstr)

    return cstr_list




def get_force_from_u_sym_in_earth_frame(vec_u, options, variables, kite, atmos, wind, architecture, parameters, *args):

    parent = architecture.parent_map[kite]

    # get relevant variables for kite n
    q = variables['x']['q' + str(kite) + str(parent)]
    dq = variables['x']['dq' + str(kite) + str(parent)]
    coeff = variables['x']['coeff' + str(kite) + str(parent)]
    wind_velocity = wind.get_velocity(q[2])

    # wind parameters
    rho_infty = atmos.get_density(q[2])

    kite_dcm = get_kite_dcm(options, variables, wind, kite, architecture, parameters)
    
    if options['wing_type'] == 'rigid_wing':
        
        Lhat = kite_dcm[:,2]

        # lift and drag coefficients
        CL = coeff[0]

        CD0 = 0.
        poss_drag_labels_in_order_of_increasing_preference = ['CX', 'CA', 'CD']
        for poss_drag_label in poss_drag_labels_in_order_of_increasing_preference:
            local_parameter_label = '[theta0,aero,' + poss_drag_label + ',0,0]'
            if local_parameter_label in parameters.labels():
                CD0 = vect_op.abs(parameters['theta0', 'aero', poss_drag_label, '0'][0])

        CD = CD0 + CL ** 2 / (np.pi * parameters['theta0', 'geometry', 'ar'])

        s_ref = parameters['theta0', 'geometry', 's_ref']

        # lift and drag force
        f_lift = CL * 1. / 2. * rho_infty * cas.mtimes(vec_u.T, vec_u) * s_ref * Lhat
        f_drag = CD * 1. / 2. * rho_infty * vect_op.norm(vec_u) * s_ref * vec_u

        f_aero = f_lift + f_drag

    if options['wing_type'] == 'LEI':

        # psi = variables['x']['psi' + str(kite) + str(parent)]
        Lhat = kite_dcm[:,2]

        CL, CD = get_aerodynamic_coefficient(get_alpha_LEI(vec_u, variables, parameters, coeff, architecture, kite))

        s_ref = parameters['theta0', 'geometry', 's_ref']

        # lift and drag force
        f_lift = CL * 1. / 2. * rho_infty * cas.mtimes(vec_u.T, vec_u) * s_ref * Lhat
        f_drag = CD * 1. / 2. * rho_infty * vect_op.norm(vec_u) * s_ref * vec_u * (1 + parameters['theta0', 'geometry', 'K_s_D'] * cas.norm_2(coeff[0]))
        f_side = np.zeros((3,))

        f_aero = f_lift + f_drag

        #f_lift = 0.5 * rho_infty * cas.mtimes(vec_u.T, vec_u) * parameters['theta0', 'geometry', 's_ref'] * CL * (cas.cross(vec_u, kite_dcm[:, 1])/cas.norm_2(cas.cross(vec_u, kite_dcm[:, 1])))
        #f_drag = 0.5 * rho_infty * cas.norm_2(vec_u) * parameters['theta0', 'geometry', 's_ref'] * CD  * (vec_u) * (1 + parameters['theta0', 'geometry', 'K_s_D'] * cas.norm_2(coeff[0]))
        # psi = 0.0
        # correction_term = (parameters['theta0', 'geometry', 'c2_s'] / cas.norm_2(vec_u)) * cas.sin(psi) * cas.cos(deg2rad(parameters['theta0', 'geometry', 'beta']))
        #correction_term = 0.0 
        #f_side = 0.5 * rho_infty * cas.mtimes(vec_u.T, vec_u) * parameters['theta0', 'geometry', 's_ref'] * parameters['theta0', 'geometry', 'A_side/A'] * parameters['theta0', 'geometry', 'c_s'] * kite_dcm[:, 1] * (coeff[0] + correction_term) 

        #f_aero =  f_lift + f_drag + f_side
        if "forces" in args:
            return f_lift, f_drag, f_side
        else:
            return f_aero
    

def get_alpha_LEI(vec_u, variables, parameters, coeff, architecture, kite):
    #coeff[1]= 0.26
    alpha_d = ((coeff[1] - parameters['theta0', 'geometry', 'u_d_0']) / (parameters['theta0', 'geometry', 'u_d_max'] - parameters['theta0', 'geometry', 'u_d_0'])) * parameters['theta0', 'geometry', 'alpha_d_max']
    # alpha = cas.arccos(cas.mtimes(vec_u.T, kite_dcm[:, 0])/ cas.norm_2(vec_u)) - deg2rad(alpha_d) + deg2rad(parameters['theta0', 'geometry', 'alpha_0'])
    # alpha =  np.arccos(cas.mtimes(vec_u.T, kite_dcm[:, 0]) / cas.norm_2(vec_u)) # - deg2rad(alpha_d) + deg2rad(parameters['theta0', 'geometry', 'alpha_0'])
    vec_t = tether_vector(variables, architecture, kite) # should be roughly "up-wards", ie, act like vec_w
    vec_v = vect_op.cross(vec_t, vec_u)
    e_x = vect_op.smooth_normalize(vect_op.cross(vec_v, vec_t))
    alpha =  np.arccos(cas.mtimes(vec_u.T, e_x) / cas.norm_2(vec_u)) - deg2rad(alpha_d) + deg2rad(parameters['theta0', 'geometry', 'alpha_0'])
    # alpha = cas.DM(np.deg2rad(15))
    return alpha

def deg2rad(angle_in_deg):
    return angle_in_deg * (cas.pi / 180)

def rad2deg(angle_in_rad):
    return angle_in_rad * (180 / cas.pi)

def get_aerodynamic_coefficient(alpha):
    """
    Calculates the aerodynamic coefficient for a given angle of attack alpha.

    :param alpha: angle of attack alpha in rad
    :return: C_l, C_D: the aerodynamic coefficient for the given AOA

    """

    alpha = rad2deg(alpha)
    # degrees = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    # CL_values = [0.1, 0.125, 0.15, 0.175, 0.2, 0.4, 0.6, 0.8, 1.0]
    # CD_values = [0.2, 0.175, 0.15, 0.125, 0.1, 0.125,0.15, 0.175, 0.2]
    # cl_f = cas.interpolant('Cl_F','bspline', [degrees], CL_values)
    # cd_f = cas.interpolant('Cd_F','bspline', [degrees], CD_values)
    # CL = cl_f(alpha)
    # CD = cd_f(alpha)

    

    lin_neg_CL= 0.0385 * alpha + 0.2542
    lin_CL_1 = 0.0641 * alpha + 0.2411
    lin_CL_2 = -0.0160 * alpha + 1.4400
    quad_CL_1 = -0.00138 * alpha **2 + 0.06375 * alpha + 0.46429
    

    # Sigmoid function for combining the linear and non-linear functions
    k1, k2, k3, k4 = 2, 2, 2, 1
    S1 = sigmoid(alpha,  0.0,  k1)     # Transition around alpha=0
    S2 = sigmoid(alpha, 12.0,  k2)     # Transition around alpha=20
    S3 = sigmoid(alpha, 40.0,  k3)     # Transition by alpha=40

    CL = (lin_neg_CL * (1 - S1) + lin_CL_1 * (S1 * (1 - S2)) + quad_CL_1 * (S2 * (1 - S3)) + lin_CL_2 * (S3))
    # S = sigmoid(alpha, 20.0,  1)     # Transition by alpha=20
    # lin_CD_1 = 0.00025 * alpha**2 + 0.1
    # lin_CD_2 = -0.00018 * alpha**2 + 0.03147 * alpha + 0.35627
    # CD = lin_CD_1 * (1 - S) + lin_CD_2 * S
    quad_neg_CD = 0.00037* alpha **2 + 0.00747 * alpha + 0.06000
    lin_pos_CD = 0.01073 * alpha + 0.05875
    S = sigmoid(alpha,  8.0,  k4)
    CD = quad_neg_CD * (1 - S) + lin_pos_CD * S

    return CL, CD

def sigmoid(alpha_sym, alpha_c, k):
    return 1.0 / (1.0 + cas.exp(-k*(alpha_sym - alpha_c)))

def get_kite_reference_frame_1p_model(tether_direction, apparent_wind_vector):
    """
    Calculates the reference frame (ex, ey, ez) for the kite.
    """
    ez =  -tether_direction / cas.norm_2(tether_direction)
    ey_cross = cas.cross(apparent_wind_vector, ez)
    ey = ey_cross / cas.norm_2(ey_cross)
    ex = cas.cross(ey, ez)
    #e_z = ez/cas.norm_2(ez)
    #e_y = ey/cas.norm_2(ey)
    #e_x = ex/cas.norm_2(ex)
    return ex, ey, ez

def tether_vector(variables, architecture, node):

    parent_map = architecture.parent_map
    parent = parent_map[node]

    q_node = struct_op.get_variable_from_model_or_reconstruction(variables, 'x', 'q' + str(node) + str(parent))

    if parent in parent_map.keys():
        grandparent = parent_map[parent]
        q_parent = struct_op.get_variable_from_model_or_reconstruction(variables, 'x', 'q' + str(parent) + str(grandparent))
    else:
        q_parent = np.zeros((3, 1))

    tether = q_node - q_parent

    return tether


def get_planar_dcm(vec_u_eff, variables, kite, architecture):

    # get relevant variables for kite n
    vec_t = tether_vector(variables, architecture, kite) # should be roughly "up-wards", ie, act like vec_w

    vec_v = vect_op.cross(vec_t, vec_u_eff)
    vec_w = vect_op.cross(vec_u_eff, vec_v)

    uhat = vect_op.smooth_normalize(vec_u_eff)
    vhat = vect_op.smooth_normalize(vec_v)
    what = vect_op.smooth_normalize(vec_w)

    planar_dcm = cas.horzcat(uhat, vhat, what)

    return planar_dcm


def get_kite_dcm(options, variables, wind, kite, architecture, parameters):

    parent = architecture.parent_map[kite]

    vec_u_eff = tools.get_u_eff_in_earth_frame(options, variables, wind, kite, architecture)

    if options['wing_type'] == 'rigid_wing':

        # roll angle
        coeff = variables['x']['coeff' + str(kite) + str(parent)]
        psi = coeff[1]

        planar_dcm = get_planar_dcm(vec_u_eff, variables, kite, architecture)
        uhat = planar_dcm[:, 0]
        vhat = planar_dcm[:, 1]
        what = planar_dcm[:, 2]

        ehat1 = uhat
        ehat2 = cas.cos(psi) * vhat + cas.sin(psi) * what
        ehat3 = cas.cos(psi) * what - cas.sin(psi) * vhat

        kite_dcm = cas.horzcat(ehat1, ehat2, ehat3)

    elif options['wing_type'] == 'LEI':

        # roll angle
        coeff = variables['x']['coeff' + str(kite) + str(parent)]
        c_s = parameters['theta0', 'geometry', 'c_s'] 
        psi = c_s * coeff[0] # u_s

        planar_dcm = get_planar_dcm(vec_u_eff, variables, kite, architecture)
        uhat = planar_dcm[:, 0]
        vhat = planar_dcm[:, 1]
        what = planar_dcm[:, 2]

        ehat1 = uhat
        ehat2 = cas.cos(psi) * vhat + cas.sin(psi) * what
        ehat3 = cas.cos(psi) * what - cas.sin(psi) * vhat

        kite_dcm = cas.horzcat(ehat1, ehat2, ehat3)

        # q = variables['x']['q' + str(kite) + str(parent)]
        # ehat1, ehat2, ehat3 =  get_kite_reference_frame_1p_model(q, vec_u_eff)
        # kite_dcm = cas.horzcat(ehat1, ehat2, ehat3)

    return kite_dcm


def get_wingtip_position(kite, options, wind, architecture, variables_si, parameters, tip):
    parent = architecture.parent_map[kite]
    q_kite = variables_si['x', 'q' + str(kite) + str(parent)]
    dcm_kite = get_kite_dcm(options, variables_si, wind, kite, architecture, parameters)
    wingtip_position = tools.construct_wingtip_position(q_kite, dcm_kite, parameters, tip)

    return wingtip_position