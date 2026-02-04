import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def value2redblue(r, s):
    """
    Generate red-blue colormap based on amplitude (r) and sign (s).
    """
    r = r * 1.7
    c = np.zeros(r.shape + (3,))
    
    c[..., 0] = (s >= 0) * r  # Red channel
    c[..., 2] = (s <= 0) * r  # Blue channel
    c[..., 1] = 1 - np.abs(r)  # Green channel for desaturation

    # Refine blending
    c[..., 2] = 1 - (s >= 0) * r
    c[..., 0] = 1 - (s <= 0) * r

    return np.clip(c, 0, 1)


def plotSHSpheres(AZI, ELE, Ysh):
    """
    Plots the amplitude and sign of spherical harmonic functions on a sphere.
    
    Parameters:
    - AZI, ELE: 2D meshgrids of azimuth and elevation angles (in radians)
    - Ysh: Matrix of spherical harmonics (numSamples x numSH)
    """
    numSh = Ysh.shape[1]
    maxOrder = int(np.sqrt(numSh) - 1)

    numAzi, numEle = AZI.shape

    r = np.abs(Ysh)
    s = np.sign(Ysh)

    # Convert spherical to Cartesian coordinates
    X = np.cos(ELE) * np.cos(AZI)
    Y = np.cos(ELE) * np.sin(AZI)
    Z = np.sin(ELE)
    
    
    s = s.reshape((numAzi, numEle, numSh))
    r = r.reshape((numAzi, numEle, numSh))

    # r = r.reshape(AZI.shape + (-1,))
    # s = s.reshape(AZI.shape + (-1,))

    iSh = 0
    fig = plt.figure(figsize=(14, 10))

    for n in range(maxOrder + 1):
        for m in range(-n, n + 1):
            ax = fig.add_subplot(maxOrder + 1, 2 * maxOrder + 1, 
                                 m + int(np.ceil(2 * maxOrder / 2)) + n * (2 * maxOrder + 1) +1,
                                 projection='3d')
            color = value2redblue(r[:, :, iSh], s[:, :, iSh])
            ax.plot_surface(X, Y, Z, facecolors=color, rstride=1, cstride=1,
                            linewidth=0, antialiased=False, shade=False)
            ax.set_box_aspect([1,1,1])
            ax.set_xticks([-1, 1])
            ax.set_yticks([-1, 1])
            ax.set_zticks([-1, 1])
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            iSh += 1

    plt.tight_layout()
    plt.show(block=False)