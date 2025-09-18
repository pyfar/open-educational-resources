import numpy as np
from VBAP_DSP import sph2cart_vec

class Scene:
    """
    Collection of dynamic properties.
    Cartesian coordinates follow the right-hand rule:
    X = forward
    Y = left
    Z = up
    """
    
    def __init__(self):
        """Initialize the default scene."""
        self.sourcePosition = [1, 2, 1]
        self.listenerPosition = [0, 0, 1]
        self.blockNumber = 0


def getDriveByScene(num_blocks):
    scenes = []
    for it_block in range(1, num_blocks + 1):
        scene = Scene()
        scene.sourcePosition[1] = (-it_block / 20 + 10)  # y-coordinate
        scene.blockNumber = it_block
        scenes.append(scene)
    return np.array(scenes)

def rotating_band_scene(in_signals, block_size):
    """
    Slowly rotating with 16 sources in the horizontal plane.
    Returns a list of Scene objects, one per block.
    """

    num_sources = in_signals.shape[1]
    az_start = np.linspace(0, (num_sources - 1) / num_sources * 2 * np.pi, num_sources)
    az_delta = 0.03 / (2 * np.pi)

    num_blocks = in_signals.shape[0] // block_size
    scene = []

    for it_block in range(num_blocks):
        az = az_start + it_block * az_delta
        pos = sph2cart_vec(np.column_stack((az, az * 0, az * 0 + 1)))  # [num_sources, 3]

        s = Scene()
        s.source_position = pos
        s.block_number = it_block + 1
        scene.append(s)

    return scene