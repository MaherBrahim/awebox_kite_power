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
import casadi.tools as cas
import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np
import awebox.tools.print_operations as print_op
import awebox.opts.kite_data.kitepower_lei_data as kitepower_lei_data
from  kite_3D_plot import plot_kite, generate_kite_wing, plot_kitepower_similar_wing, animate_3d_flight

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
options['model.model_bounds.tether_stress.include']  = True
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
trial.optimize(final_homotopy_step = 'initial_guess')  # 'initial_guess', 'initial', 'fictitious', 'power', 'final'



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

# kite reference frame
e_x = plot_dict['outputs']['aerodynamics']['e_x1']
e_y = plot_dict['outputs']['aerodynamics']['e_y1']
e_z = plot_dict['outputs']['aerodynamics']['e_z1']
import pdb; pdb.set_trace()
# aerodynamic forces
lift_force = outputs['aerodynamics']['F_lift_LEI_Kite1']
drag_force = outputs['aerodynamics']['F_drag_LEI_Kite1']
side_force = outputs['aerodynamics']['F_side_LEI_Kite1']

def get_norm(vector_input):
    x, y, z = vector_input
    return np.array([np.sqrt(x[i]**2 + y[i]**2 + z[i]**2) for i in range(len(x))])

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
plt.plot(time, (180/cas.pi) * outputs['aerodynamics']['alpha1'][0], label='Angle of Attack')
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


alpha_sim = (180/cas.pi) * outputs['aerodynamics']['alpha1'][0] 
alpha = np.linspace(-20, 90, 50)
CL_measured = outputs['aerodynamics']['CL_LEI_Kite1'][0]
CD_measured = outputs['aerodynamics']['CD_LEI_Kite1'][0]



lin_neg_CL= 0.0385 * alpha + 0.2542
lin_CL_1 = 0.0641 * alpha + 0.2411
lin_CL_2 = -0.0160 * alpha + 1.4400
quad_CL_1 = -0.00138 * alpha **2 + 0.06375 * alpha + 0.46429
def sigmoid(alpha_sym, alpha_c, k):
    return 1.0 / (1.0 + cas.exp(-k*(alpha_sym - alpha_c)))
# Sigmoid function for combining the two linear functions
k1, k2, k3, k4 = 0.5, 0.5, 0.5, 1
S1 = sigmoid(alpha,  0.0,  k1)     # Transition around alpha=0
S2 = sigmoid(alpha, 12.0,  k2)     # Transition around alpha=12
S3 = sigmoid(alpha, 40.0,  k3)     # Transition by alpha=40
CL_fitted = (lin_neg_CL * (1 - S1) + lin_CL_1 * (S1 * (1 - S2)) + quad_CL_1 * (S2 * (1 - S3)) + lin_CL_2 * (S3))

quad_neg_CD = 0.00037* alpha **2 + 0.00747 * alpha + 0.06000
lin_pos_CD = 0.01073 * alpha + 0.05875
S = sigmoid(alpha,  8.0,  k4)
CD_fitted = quad_neg_CD * (1 - S) + lin_pos_CD * S


plt.figure()
plt.scatter(alpha_sim, CL_measured, label='C_L-values used in the simulation', alpha=0.6)
plt.scatter(alpha_sim, CD_measured, label='C_D-values used in the simulation', alpha=0.6)
# print('alpha_sim:', alpha_sim)
# print('CL_measured', CL_measured)
# print('CD_measured', CD_measured)

plt.plot(alpha, CL_fitted, label='Fitted C_L', linewidth=2)
plt.plot(alpha, CD_fitted, label='Fitted C_D', linewidth=2)

plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'Coefficients $C_L$ and $C_D$')
plt.legend()
plt.grid(True)
plt.title('Aerodynamic Coefficient Fitting')


plt.figure() 
plt.plot(time, plot_dict['x']['coeff10'][0], label = 'u_s')
plt.plot(time, plot_dict['x']['coeff10'][1], label = 'u_d')
plt.xlabel('t[s]')
plt.ylabel('u_d and u_s[-]')
plt.legend()
plt.grid(True)

plt.figure() 
plt.plot(time, get_norm(lift_force), label = 'F_L')
plt.plot(time, get_norm(drag_force), label = 'F_D')
plt.plot(time, get_norm(side_force), label = 'F_S')
plt.xlabel('t[s]')
plt.ylabel('Force [N]')
plt.legend()
plt.grid(True)


plot_kite(kite_positions,  e_y, e_x, e_z, kite_size=1.5)

# Parameters of the kite wing
w = 5.77               # width 
h = 2                  # Depth of each segment
curve_height = 2.23    # Maximum height of the curvature
num_segments = 5      # Number of panels

panels = generate_kite_wing(w, h, curve_height, num_segments)
plot_kitepower_similar_wing(panels, kite_positions, e_y, e_x, e_z)



# animate_3d_flight(kite_positions, [lift_force, drag_force, side_force], force_labels=["Lift Force", "Drag Force", "Side Force"])


# animate_3d_flight(kite_positions, [e_x, e_y, e_z], force_labels=["e_x", "e_y", "e_z"])

wind = plot_dict['outputs']['aerodynamics']['u_infty1']
apparent_wind = plot_dict['outputs']['aerodynamics']['vec_u1']
kite_vel = plot_dict['x']['dq10']
true_apparent_wind = plot_dict['outputs']['aerodynamics']['true_vec_u1']
# animate_3d_flight(kite_positions, [e_x, wind, kite_vel, apparent_wind, true_apparent_wind], force_labels=["e_x", "wind", "kite_vel", "apparent wind", "true_apparent_wind"])

plt.show()

