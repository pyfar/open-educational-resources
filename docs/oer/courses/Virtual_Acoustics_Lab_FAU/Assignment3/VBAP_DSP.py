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

    def show_hull(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for simplex in self.hull:
            tri = self.ls_positions[simplex]
            ax.plot_trisurf(tri[:, 0], tri[:, 1], tri[:, 2], alpha=0.5)
        ax.scatter(*self.ls_positions.T, color='yellow', edgecolors='black', s=50)
        for i, (x, y, z) in enumerate(self.ls_positions):
            ax.text(x, y, z, str(i+1), fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title("Loudspeaker Hull")
        ax.grid(True)
        plt.axis('equal')
        plt.show()

    def set_sources(self, virtual_azi, virtual_ele):
        assert virtual_azi.shape == virtual_ele.shape
        num_src = virtual_azi.shape[0]
        src_pos = sph2cart_vec(np.stack((virtual_azi, virtual_ele, np.ones(num_src)), axis=-1))
        self.ls_gains_new = self.calculate_gains(src_pos)

    def setSources(self, virtual_azi, virtual_ele):
        return self.set_sources(virtual_azi, virtual_ele)

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

    def calculate_gains(self, src_pos):
        """
        Placeholder: For each source position, find the closest loudspeaker and set its gain to 1 (all others 0).
        This should be replaced with a proper VBAP gain calculation for 3D arrays.
        Args:
            src_pos: [num_src, 3] array of source positions (cartesian)
        Returns:
            gains: [num_src, num_ls] array of gains
        """
        num_src = src_pos.shape[0]
        gains = np.zeros((num_src, self.num_ls))
        for i in range(num_src):
            dists = np.linalg.norm(self.ls_positions - src_pos[i, :], axis=1)
            idx = np.argmin(dists)
            gains[i, idx] = 1.0
        return gains