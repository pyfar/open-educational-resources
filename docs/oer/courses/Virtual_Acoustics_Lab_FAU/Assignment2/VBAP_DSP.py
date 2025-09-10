import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def sph2cart_vec(sph):
    # sph = [azimuth, elevation, radius] in radians
    azimuth = sph[:, 0]
    elevation = sph[:, 1]
    radius = sph[:, 2]
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    return np.stack((x, y, z), axis=-1)

class VBAP_DSP:
    def __init__(self, ls_positions):
        assert ls_positions.shape[1] == 3
        self.ls_positions = ls_positions
        self.num_ls = ls_positions.shape[0]
        self.ls_gains_new = None
        self.ls_gains_last = None
        self.hull = None
        self.ax = None

    def get_conv_hull(self):
        hull = ConvexHull(self.ls_positions)
        self.hull = hull.simplices
        return hull.simplices

    def show_hull(self):
        fig = plt.figure(figsize=(12, 8))  # Increased figure size
        ax = fig.add_subplot(111, projection='3d')
        for simplex in self.hull:
            tri = self.ls_positions[simplex]
            # Plot the filled surface
            ax.plot_trisurf(tri[:, 0], tri[:, 1], tri[:, 2], alpha=0.5)
            ax.scatter(*self.ls_positions.T, color='yellow', s=50)
            # Plot the edges
            for i in range(3):
                ax.plot([tri[i,0], tri[(i+1)%3,0]], 
                       [tri[i,1], tri[(i+1)%3,1]], 
                       [tri[i,2], tri[(i+1)%3,2]], 
                       color='gray', 
                       linewidth=0.7,
                       zorder=2)
        # Plot text labels with highest z-order, positioned outside the hull
        for i, (x, y, z) in enumerate(self.ls_positions):
            ax.text(x, y, z, str(i+1), fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title("Loudspeaker Hull")
        ax.grid(True)
        plt.axis('equal')
        # Set initial view angle
        ax.view_init(azim=45 +180)
        self.ax = ax
        return ax

    def set_sources(self, virtual_azi, virtual_ele):
        assert virtual_azi.shape == virtual_ele.shape
        num_src = virtual_azi.shape[0]
        src_pos = sph2cart_vec(np.stack((virtual_azi, virtual_ele, np.ones(num_src)), axis=-1))
        self.ls_gains_new = self.calculate_gains(src_pos)

    def current_ls_gains(self):
        if self.ls_gains_last is None or self.ls_gains_last.shape != self.ls_gains_new.shape:
            self.ls_gains_last = self.ls_gains_new
        ls_gains = 0.5 * self.ls_gains_new + 0.5 * self.ls_gains_last
        self.ls_gains_last = self.ls_gains_new
        return ls_gains

    def process(self, input_sig):
        # input_sig shape: [t, num_src]
        ls_gains = self.current_ls_gains()  # shape: [num_src, num_ls]
        output_sig = input_sig @ ls_gains  # shape: [t, num_ls]
        return output_sig