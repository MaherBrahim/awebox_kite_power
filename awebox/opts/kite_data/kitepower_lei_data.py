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
import numpy as np
from casadi.tools import vertcat

def data_dict():

    data_dict = {}
    data_dict['name'] = 'kitepower_lei'

    data_dict['geometry'] = geometry() # kite geometry

    stab_derivs, aero_validity = aero()
    data_dict['stab_derivs'] = stab_derivs # stability derivatives
    data_dict['aero_validity'] = aero_validity

    return data_dict

def geometry():

    geometry = {}
    geometry['b_ref'] = 5 # fill in some meaningful value
    geometry['s_ref'] = 10.18  # [m^2]
    geometry['c_ref'] = geometry['s_ref']  / geometry['b_ref']  # [m]
    
    # TODO: add all relevant parameters here
    geometry['ar'] = 10.0 # can be deleted later
 
    # kite mass + KCU ?
    geometry['m_k'] = 6.21 + 8.4  # [kg]

    # tether attachment point
    geometry['r_tether'] = np.zeros((3,1))

    # steering coefficient
    geometry['c_s'] = 2.59 # [-]

    # correction factor
    geometry['c2_s'] = 0.93 # [-]

    # Relative side area 
    geometry['A_side/A'] = 0.306 # [-]

    # Straight tether elevation angle
    geometry['beta'] = 74.7 # [deg]

    # Depower angle offset 
    geometry['alpha_0'] = 0.0 # [deg]

    # Depower angle 
    geometry['alpha_d_max'] = 31.0 # [deg]

    # Depower offset
    geometry['u_d_0'] = 0.231

    # Max depower setting
    geometry['u_d_max'] = 0.4247

    # Steering-induced drag coefficient 
    geometry['K_s_D'] = 0.6 # [-]

    # Steering offset c0 
    geometry['c0'] = -0.004 # [-]

    # Steering constant c1
    geometry['c1'] = 0.264  # [rad/m]

    # Steering constant c2
    geometry['c2'] = 6.20  # [rad m/s^2]

    return geometry


def aero():
    # commented values are not currently supported, future implementation

    # A reference model for airborne wind energy systems for optimization and control
    # Article
    # March 2019 Renewable Energy
    # Elena Malz Jonas Koenemann S. Sieberling Sebastien Gros

    # commented values are not currently supported, future implementation

    stab_derivs = {}
    aero_validity = {}

    aero_validity['alpha_max_deg'] = 20.0
    aero_validity['alpha_min_deg'] = -20.0
    aero_validity['beta_max_deg'] = 20.0
    aero_validity['beta_min_deg'] = -20.0
    return stab_derivs, aero_validity
