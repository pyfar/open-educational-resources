import numpy as np

def cart2sph_vec(cart):
    """
    Vectorized conversion from Cartesian to spherical coordinates.
    Supports both 2D (x, y) and 3D (x, y, z) input.
    
    Args:
        cart (np.ndarray): [N, 2] or [N, 3] array of [x, y] or [x, y, z]
                          or [2] or [3] array for single point
        
    Returns:
        np.ndarray: [N, 3] array of [azimuth, elevation, radius] or [3] array for single point
    """
    # Ensure input is 2D array
    cart = np.asarray(cart)
    if cart.ndim == 1:
        cart = cart.reshape(1, -1)
    
    x = cart[:, 0]
    y = cart[:, 1]
    
    # Handle 2D case (no z-coordinate)
    if cart.shape[1] == 2:
        z = np.zeros_like(x)
    else:
        z = cart[:, 2]

    az = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        elev = np.arcsin(z / r)
        elev = np.where(r == 0, 0, elev)

    sph = np.column_stack((az, elev, r))
    
    return sph


def sph2cart_vec(sph):
    """
    Vectorized conversion from spherical to Cartesian coordinates.
    
    Args:
        sph (np.ndarray): [N, 3] array of [azimuth, elevation, radius] or [3] array for single point
        
    Returns:
        np.ndarray: [N, 3] array of [x, y, z] or [3] array for single point
    """
    # Ensure input is 2D array
    sph = np.asarray(sph)
    if sph.ndim == 1:
        sph = sph.reshape(1, -1)
    
    azi = sph[:, 0]
    ele = sph[:, 1]
    r = sph[:, 2]

    x = r * np.cos(ele) * np.cos(azi)
    y = r * np.cos(ele) * np.sin(azi)
    z = r * np.sin(ele)

    cart = np.column_stack((x, y, z))
    
    return cart 