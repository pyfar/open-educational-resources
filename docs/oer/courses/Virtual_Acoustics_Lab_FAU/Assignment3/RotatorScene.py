import numpy as np

def rotator_scene():
    """
    Creates a scene with a source that spins around the origin once.
    
    Returns:
        tuple: (scene, num_blocks)
            - scene: list of dictionaries containing source and listener positions
            - num_blocks: number of blocks in the scene
    """
    num_blocks = 1500
    
    # Create source positions using spherical to cartesian conversion
    # Source moves in a circle around the listener at constant elevation
    azimuth = np.linspace(0, 2*np.pi, num_blocks)
    elevation = np.zeros(num_blocks)
    radius = np.ones(num_blocks)
    
    # Convert spherical to cartesian coordinates
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    
    # Create scene list
    scene = []
    for i in range(num_blocks):
        scene.append({
            'sourcePosition': np.array([x[i], y[i], z[i]]),
            'listenerPosition': np.array([0, 0, 0])
        })
    
    return scene, num_blocks 