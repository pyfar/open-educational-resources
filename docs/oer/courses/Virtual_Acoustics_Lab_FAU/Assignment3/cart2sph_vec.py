import numpy as np

def cart2sph_vec(cart):
    """
    Vectorized conversion from Cartesian to spherical coordinates.
    Input:
        cart : [N, 3] array of [x, y, z] or [3] array for single point
    Output:
        sph : [N, 3] array of [azimuth, elevation, radius] or [3] array for single point
    """
    # Ensure input is 2D array
    cart = np.asarray(cart)
    if cart.ndim == 1:
        cart = cart.reshape(1, -1)
    
    x = cart[:, 0]
    y = cart[:, 1]
    z = cart[:, 2]

    az = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2 + z**2)
    elev = np.arcsin(z / r)

    sph = np.column_stack((az, elev, r))
    
    return sph