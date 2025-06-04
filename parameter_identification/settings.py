import pathlib
import json
from functools import lru_cache


# Settings the directory for the HSL library
dll_dir = pathlib.Path(r"C:\Users\maher\OneDrive\Desktop\Masterarbeit\Code2.0\awebox_kite_power\toolchain\bin")


# Base directory for data files
BASE_DIR = pathlib.Path(__file__).parent.parent.parent / "Data" / "DataShots" / "2024-09-24_13-49-14"
MEAS_FILE = "one_loop_meas_2024-09-24_6630_6660.json"

# settings for the collocation method
def default_collocation_opts():
    return {
        # Number of collocation stages (ns)
        'ns': 1,
        # Number of finite elements per interval (N_fe)
        'N_fe': 1,
        # Number of the used measurements
        'N': 101,
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

def load_measurement_data(filename: str) -> dict:
    """
    Load measurement data from a JSON file.
    Args:
        filename (str): The name of the JSON file containing the measurement data.
    """
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Measurement file {filename} does not exist at {path}.")
    
    with open(path, "r") as f:
        data = json.load(f)
    return data