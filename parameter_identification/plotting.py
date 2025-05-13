import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines  
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
import matplotlib.colors as mcolors

def plot_kite(positions, ex_array, ey_array, ez_array, kite_size):
    """
    Plots a kite with positions and orientation vectors in 3D space 'Triangle shaped kite'.
    """
    num_states = len(ex_array[0])
    ground_point = np.array([0, 0, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(num_states):
        center = np.array([positions[0][i], positions[1][i], positions[2][i]])
        ex = np.array([ex_array[0][i], ex_array[1][i], ex_array[2][i]])
        ey = np.array([ey_array[0][i], ey_array[1][i], ey_array[2][i]])
        ez = np.array([-ez_array[0][i], -ez_array[1][i], -ez_array[2][i]])

        ex /= np.linalg.norm(ex)
        ey /= np.linalg.norm(ey)
        ez /= np.linalg.norm(ez)

        kite_shape_local = np.array([[0, 1], [2, -1], [-2, -1], [0, -0.5]]) * kite_size
        kite_shape_global = (
            kite_shape_local[:, 0][:, np.newaxis] * ex +
            kite_shape_local[:, 1][:, np.newaxis] * ey
        ) + center

        A, B, C, D = kite_shape_global

        ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], 'b-', alpha=0.5)
        ax.plot([A[0], C[0]], [A[1], C[1]], [A[2], C[2]], 'b-', alpha=0.5)
        ax.plot([D[0], B[0]], [D[1], B[1]], [D[2], B[2]], 'b-', alpha=0.5)
        ax.plot([D[0], C[0]], [D[1], C[1]], [D[2], C[2]], 'b-', alpha=0.5)

        verts = [[A, B, D, C]]
        ax.add_collection3d(Poly3DCollection(verts, color='gray', alpha=0.4))

        ax.plot([ground_point[0], center[0]],
                [ground_point[1], center[1]],
                [ground_point[2], center[2]],
                'black', linewidth=0.7, alpha=0.3)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    plt.title("Kite states plotted with triangular shape ")



def generate_kite_wing(w, h, curve_height, num_segments):
    """
    Generate kite wing panels with specified curvature
    """
    # Points for the curvature
    x = np.linspace(-w / 2, w / 2, num_segments + 1)
    z = curve_height * (1 - (2 * x / w) ** 2)  

   
    panels = []
    for i in range(num_segments):
        if i == 0:  # First panel
            p1 = np.array([x[i], -h / 2, z[i]])         # Bottom left
            p2 = np.array([x[i + 1], 0, z[i + 1]])      # Bottom right
            p3 = np.array([x[i + 1], -h, z[i + 1]])     # Top right
            p4 = np.array([x[i], -h / 2, z[i]])         # Top left
        elif i == num_segments - 1:  # Last panel
            p1 = np.array([x[i], 0, z[i]])              # Unten links
            p2 = np.array([x[i + 1], -h / 2, z[i + 1]]) # Unten rechts
            p3 = np.array([x[i + 1], -h / 2, z[i + 1]]) # Oben rechts
            p4 = np.array([x[i], -h, z[i]])             # Oben links
        else:   
            p1 = np.array([x[i], 0, z[i]])              
            p2 = np.array([x[i + 1], 0, z[i + 1]])      
            p3 = np.array([x[i + 1], -h, z[i + 1]])     
            p4 = np.array([x[i], -h, z[i]])             

        panels.append([p1, p2, p3, p4])

    return panels

def plot_kitepower_similar_wing(panels, positions, ex_array, ey_array, ez_array):
    """
    Plot the curved kite wing in 3D and add ropes. 
    The shape is similar to the model used by Kitepower.
    """
    num_states = len(ex_array[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(num_states):
        # Extract position and orientation
        center = np.array([positions[0][i], positions[1][i], positions[2][i]])
        ex = np.array([ex_array[0][i], ex_array[1][i], ex_array[2][i]])
        ey = np.array([ey_array[0][i], ey_array[1][i], ey_array[2][i]])
        ez = np.array([-ez_array[0][i], -ez_array[1][i], -ez_array[2][i]])

        # Normalise vectors
        ex = ex / np.linalg.norm(ex)
        ey = ey / np.linalg.norm(ey)
        ez = ez / np.linalg.norm(ez)

        # Plot panels with orientation
        for panel in panels:
            panel_rotated = [(point[0] * ex + point[1] * ey + point[2] * ez) + center for point in panel]
            poly = Poly3DCollection([panel_rotated], alpha=0.6, edgecolor='k')
            poly.set_facecolor('lightblue')
            ax.add_collection3d(poly)

        # Add ropes
        left_point_lower = panels[0][0]    # Lower left corner of the first panel
        right_point_lower = panels[-1][1]  # Lower right corner of the last panel

        # Transform the rope points with the kite orientation
        left_point_lower_rotated = (left_point_lower[0] * ex + left_point_lower[1] * ey + left_point_lower[2] * ez) + center
        right_point_lower_rotated = (right_point_lower[0] * ex + right_point_lower[1] * ey + right_point_lower[2] * ez) + center

        # Add ropes to the central point 
        ax.plot([left_point_lower_rotated[0], center[0]], [left_point_lower_rotated[1], center[1]], [left_point_lower_rotated[2], center[2]], color='gray', linewidth=1)
        ax.plot([right_point_lower_rotated[0], center[0]], [right_point_lower_rotated[1], center[1]], [right_point_lower_rotated[2], center[2]], color='gray', linewidth=1)

        # Add line to the floor
        ground_point = np.array([0, 0, 0])
        ax.plot([ground_point[0], center[0]], [ground_point[1], center[1]], [ground_point[2], center[2]], 'black', linewidth=0.7, alpha=0.3)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D kite wing with rope connections")

# test example
positions = [
    np.linspace(0, 10, 10),  
    np.linspace(0, -10, 10), 
    np.linspace(0, 50, 10)   
]
ex_array = [
    np.ones(10), 
    np.zeros(10),
    np.zeros(10)
]
ey_array = [
    np.zeros(10),
    np.ones(10), 
    np.zeros(10)
]
ez_array = [
    np.zeros(10),
    np.zeros(10),
    np.ones(10)  
]

# Parameters of the kite wing
w = 5.77               # width 
h = 2                  # Depth of each segment
curve_height = 2.23    # Maximum height of the curvature
num_segments = 10      # Number of panels
#panels = generate_kite_wing(w, h, curve_height, num_segments)
#plot_kitepower_similar_wing(panels, positions, ex_array, ey_array, ez_array)


def draw_lightning(ax, position, color, scale):
    points = [(0, 0), (1, 3), (0.5, 3), (1.5, 6), (0, 4), (0.5, 4)]

    points3d = [((x * scale) + position[0], (y * scale) + position[1], position[2]) for (x, y) in points]

    poly = Poly3DCollection([points3d], facecolors=color, edgecolors='black', linewidths=1, alpha=0.7)
    ax.add_collection3d(poly)

def animate_3d_flight(positions, forces, force_labels):
    """
    Creates a 3D animation of a flight path including forces.
    """
    pastel_colors = ['#179C7D', '#F58220', '#A6BBC8', '#FF5733', '#4A90E2']  # Extendable color palette
    
    pos = np.array(positions)
    force_vectors = [np.array(f) for f in forces]
    n_points = pos.shape[1]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    
    # Main flight path, current point, and ground line
    line, = ax.plot([], [], [], 'gray', lw=1, alpha=0.5, label='Flight Path')
    point, = ax.plot([], [], [], 'ro', markersize=5, alpha=0.8, label='Object')
    ground_line, = ax.plot([], [], [], lw=1, alpha=0.8, label='Tether')
    
    # Quivers (forces) - to be updated dynamically
    quivers = [None] * len(forces)

    lightning_size = 2
    threshold = 200
    lightning_art = [None]
    
    # Determine axis limits
    x_min, x_max = pos[0].min(), pos[0].max()
    y_min, y_max = pos[1].min(), pos[1].max()
    z_min, z_max = pos[2].min(), pos[2].max()
    dx, dy, dz = x_max, y_max, z_max  # Can be adjusted for better scaling
    
    # Create dummy lines for forces to show in legend
    force_legends = [mlines.Line2D([], [], color=color, label=label) for color, label in zip(pastel_colors, force_labels)]
    
    def init():
        ax.set_xlim3d([x_min - dx, x_max + dx])
        ax.set_ylim3d([y_min - dy, y_max + dy])
        ax.set_zlim3d([z_min - dz, z_max + dz])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        ground_line.set_data([], [])
        ground_line.set_3d_properties([])
        
        # Add legend (includes dummy lines for forces)
        ax.legend(handles=[line, ground_line] + force_legends)
        
        return [line, point, ground_line]
    
    def update(frame):
        nonlocal quivers
        
        # Flight path up to current frame
        line.set_data(pos[0][:frame], pos[1][:frame])
        line.set_3d_properties(pos[2][:frame])
        
        # Current point
        point.set_data(pos[0][frame], pos[1][frame])
        point.set_3d_properties(pos[2][frame])
        
        # Ground line from (0,0,0) to current position
        ground_line.set_data([0, pos[0][frame]], [0, pos[1][frame]])
        ground_line.set_3d_properties([0, pos[2][frame]])
        
        # Remove old quivers
        for q in quivers:
            if q:
                q.remove()
        
        # Draw new quivers (forces)
        arrow_length = 100
        quivers = [ax.quiver(pos[0][frame], pos[1][frame], pos[2][frame],
                              force[0][frame], force[1][frame], force[2][frame],
                              color=color, length=arrow_length, normalize=True)
                   for force, color in zip(force_vectors, pastel_colors)]
        
        kite_pos = np.array([pos[0][frame], pos[1][frame], pos[2][frame]])
        distance = np.linalg.norm(kite_pos)
        lightning_color = 'yellow' if distance >= threshold else 'red'
        if lightning_art[0] is not None:
            lightning_art[0].remove()
        lightning_art[0] = draw_lightning(ax, (2 * lightning_size, 0, 0), lightning_color, lightning_size)
        return [line, point, ground_line] + quivers + [lightning_art[0]]
    
    anim = FuncAnimation(fig, update, frames=n_points, init_func=init,
                         blit=False, interval=200)
    plt.show()
    return anim

def plot_xy(x, y_series, labels, xlabel='X-Achse', ylabel='Y-Achse', title='XY Plot', figsize=(10,6)):
    fig, ax = plt.subplots(figsize=figsize)
    for y, label in zip(y_series, labels):
        ax.plot(x, y, label=label, marker='.', markersize=4, linestyle='-')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig, ax

def plot_xyz(x, y, z, xlabel='X-Achse', ylabel='Y-Achse', zlabel='Z-Achse', title='XYZ Plot'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, marker='.', markersize=4, linestyle='-')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.grid(True)
    return fig, ax

import matplotlib.pyplot as plt

def plot_xy_mixed(x_list, y_groups, labels_groups,
                  xlabel='x', ylabel='y', title='', pastel_alpha=0.5):
    """
    Plot groups of y-series against their respective x-axis on a single plot.
    """
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots()
    
    for group_idx, (x, y_list, labels) in enumerate(zip(x_list, y_groups, labels_groups)):
        for i, (y, lbl) in enumerate(zip(y_list, labels)):
            color = mcolors.to_rgba(base_colors[i % len(base_colors)])
            if group_idx == 1:
                # pastel = blend with white
                pastel_color = tuple(1 - (1 - c)*pastel_alpha for c in color[:3]) + (color[3],)
                ax.plot(x, y, marker='.', linestyle='-', label=lbl, color=pastel_color)
            else:
                ax.plot(x, y, marker='.', linestyle='-', label=lbl, color=color)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return fig, ax


def is_gaussian_noise(
    data: np.ndarray,
    window_length: int = 21,
    polyorder: int = 5,
    bins: int = 30):

        """
        Smooth data with a Savitzky-Golay filter, compute residual noise, estimate
        μ, σ², σ, and plot (1) raw vs. smoothed signal, (2) noise histogram with a
        fitted Gaussian PDF.

        Returns (mu, var, sigma).
        """
        data = np.asarray(data, dtype=float)

        if window_length % 2 == 0 or window_length < polyorder + 2:
            raise ValueError("window_length must be odd and > polyorder + 1")
        if window_length > data.size:
            raise ValueError("window_length exceeds data length")

        smooth = savgol_filter(data, window_length, polyorder)
        residuals = data - smooth

        mu = residuals.mean()
        var = residuals.var(ddof=1)
        sigma = np.sqrt(var)

        # Plot 1: original vs. smoothed
        plt.figure()
        plt.plot(data,   label="Original",  lw=1)
        plt.plot(smooth, label="Smoothed",  lw=2)
        plt.title("Signal vs. Savitzky–Golay")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Plot 2: noise histogram + Gaussian PDF
        plt.figure()
        plt.hist(residuals, bins=bins, density=True, alpha=0.5, label="Residuals")
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 800)
        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        plt.plot(x, pdf, lw=2, label=f"Gaussian μ={mu:.3f}, σ={sigma:.3f}")
        plt.title("Noise Distribution vs. Gaussian")
        plt.xlabel("Residual")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return mu, var, sigma
