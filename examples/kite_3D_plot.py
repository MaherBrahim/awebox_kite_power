import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
        ez = np.array([ez_array[0][i], ez_array[1][i], ez_array[2][i]])

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
    # Points for the curvature
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
        ez = np.array([ez_array[0][i], ez_array[1][i], ez_array[2][i]])

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
