import numpy as np

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