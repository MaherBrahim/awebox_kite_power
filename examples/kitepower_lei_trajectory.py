#!/usr/bin/python3
"""
Circular pumping trajectory for the Ampyx AP2 aircraft.
Model and constraints as in:

"Performance assessment of a rigid wing Airborne Wind Energy pumping system",
G. Licitra, J. Koenemann, A. Bürger, P. Williams, R. Ruiterkamp, M. Diehl
Energy, Vol.173, pp. 569-585, 2019.

:author: Jochem De Schutter
:edited: Rachel Leuthold
"""

import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np
import awebox.tools.print_operations as print_op
import awebox.opts.kite_data.kitepower_lei_data as kitepower_lei_data
from  kite_3D_plot import plot_kite, generate_kite_wing, plot_kitepower_similar_wing

# indicate desired system architecture
# here: single kite with 6DOF Ampyx AP2 model
options = {}
options['user_options.system_model.architecture'] = {1: 0}
options['user_options.kite_standard'] = kitepower_lei_data.data_dict()
options['user_options.system_model.wing_type'] = 'LEI'
options['user_options.system_model.kite_dof'] = 3

# indicate desired operation mode
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
windings = 1
options['user_options.trajectory.lift_mode.windings'] = windings

# indicate desired environment
options['params.wind.z_ref'] = 100.0
options['params.wind.power_wind.exp_ref'] = 0.15
options['user_options.wind.model'] = 'power'
options['user_options.wind.u_ref'] = 10.

# coefficient boundaries
options['model.system_bounds.x.coeff'] =  [np.array([-1., 0.]), np.array([1., 1.])]

# indicate numerical nlp details
# here: nlp discretization, with a zero-order-hold control parametrization, and
# a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps
# within ipopt.
options['nlp.n_k'] = int(40/3 * windings)
options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple' # 'single_reelout'
options['solver.linear_solver'] = 'mumps'  # if HSL is installed, otherwise 'mumps'
options['model.system_bounds.x.ddl_t'] = [-2.0, 2.0]
options['model.system_bounds.theta.t_f'] = [0.0, windings*30.0]

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

# build and optimize the NLP (trial)
trial = awe.Trial(options, 'Kitepower_LEI')
trial.build()
trial.optimize(final_homotopy_step = 'final')



# write the solution to CSV file, interpolating the collocation solution with given frequency.
# trial.write_to_csv(filename = 'Ampyx_AP2_solution', frequency = 30)

# draw some of the pre-coded plots for analysis
trial.plot(['isometric', 'states', 'controls', 'constraints'])


# extract information from the solution for independent plotting or post-processing
# here: plot relevant system outputs, compare to [Licitra2019, Fig 11].
plot_dict = trial.visualization.plot_dict
outputs = plot_dict['outputs']
time = plot_dict['time_grids']['ip']
avg_power = plot_dict['power_and_performance']['avg_power']/1e3
kite_positions = plot_dict['x']['q10']
e_radial =plot_dict['outputs']['rotation']['ehat_radial1']
e_tngential = plot_dict['outputs']['rotation']['ehat_tangential1']
e_normal = plot_dict['outputs']['rotation']['ehat_normal0']



print('======================================')
print('Average power: {} kW'.format(avg_power))
print('======================================')

plt.subplots(6, 1, sharex=True)
plt.subplot(611)
plt.plot(time, plot_dict['x']['l_t'][0], label='Tether Length')
plt.ylabel('[m]')
plt.legend()
plt.grid(True)

plt.subplot(612)
plt.plot(time, plot_dict['x']['dl_t'][0], label='Tether Reel-out Speed')
plt.ylabel('[m/s]')
plt.legend()
plt.hlines([20, -15], time[0], time[-1], linestyle='--', color='black')
plt.grid(True)

plt.subplot(613)
plt.plot(time, outputs['aerodynamics']['airspeed1'][0], label='Airspeed')
plt.ylabel('[m/s]')
plt.legend()
plt.grid(True)

plt.subplot(614)
plt.plot(time, outputs['aerodynamics']['alpha1'][0], label='Angle of Attack')
plt.ylabel('[deg]')
plt.legend()
plt.hlines([20, -20], time[0], time[-1], linestyle='--', color='black')
plt.grid(True)

plt.subplot(615)
plt.plot(time, outputs['local_performance']['tether_force10'][0], label='Tether Force Magnitude')
plt.ylabel('[N]')
plt.xlabel('t [s]')
plt.legend()
plt.grid(True)

plt.subplot(616)
plt.plot(time, outputs['aerodynamics']['f_aero_earth1'][0], label='Aero Force X')
plt.ylabel('[N]')
plt.xlabel('t [s]')
plt.legend()
plt.grid(True)

plt.figure() 
plt.plot(time, plot_dict['x']['coeff10'][0], label = 'u_s')
plt.plot(time, plot_dict['x']['coeff10'][1], label = 'u_d')
plt.xlabel('t[s]')
plt.ylabel('u_d and u_s[-]')
plt.legend()
plt.grid(True)

plot_kite(kite_positions,  e_radial, e_tngential, e_normal, kite_size=1.5)

# Parameters of the kite wing
w = 5.77               # width 
h = 2                  # Depth of each segment
curve_height = 2.23    # Maximum height of the curvature
num_segments = 5      # Number of panels

panels = generate_kite_wing(w, h, curve_height, num_segments)
plot_kitepower_similar_wing(panels, kite_positions, e_radial, e_tngential, e_normal)

plt.show()
