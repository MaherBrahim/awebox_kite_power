import pathlib
dll_dir = pathlib.Path(r"C:\Users\maher\OneDrive\Desktop\Masterarbeit\Code2.0\awebox_kite_power\toolchain\bin")
def default_collocation_opts():
    return {
        # Number of collocation stages (ns)
        'ns': 1,
        # Number of finite elements per interval (N_fe)
        'N_fe': 1,
        # Number of the used measurements
        'N': 10,
    }

# Solver parameters
def default_solver_opts():
    return {
        'ipopt': {
            # Linear solver for KKT system
            'linear_solver': 'ma57',
            # Path to the HSL library for MA57
            'hsllib': str(dll_dir / "libhsl.dll"),  
        }
    }

# Plotting parameters
def default_plot_opts():
    return {
        'show': True,    
        'save': False,   
        'format': 'png', 
        'dpi': 150,      
    }

# Aggregate all defaults
def default_options():
    return {
        'collocation': default_collocation_opts(),
        'solver':      default_solver_opts(),
        'plot':        default_plot_opts(),
    }
