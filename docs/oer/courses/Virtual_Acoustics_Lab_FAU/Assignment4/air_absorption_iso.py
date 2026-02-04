import numpy as np

def air_absorption_iso(f, T, hr, ps=None):
    """
    Calculate sound absorption (attenuation) in humid air using ISO standard.
    
    Args:
        f (np.ndarray): Frequency in Hz
        T (float): Temperature in degrees Celsius
        hr (float): Relative humidity in percent (0-100)
        ps (float, optional): Atmospheric pressure ratio, default is 1
        
    Returns:
        tuple: (alpha_iso, c_iso) where alpha_iso is absorption in dB/m and c_iso is speed of sound
    """
    if ps is None:
        ps = 1
    
    # Convert T from Celsius to Kelvin
    T = 273.15 + T
    
    # Constants
    T01 = 273.16  # triple point in degrees Kelvin
    T0 = 293.15
    
    # Atmospheric pressure ratio
    ps0 = 1  # ps0 = standard pressure/standard pressure which is unity
    
    # ISO formula for saturation pressure ratio
    psat_ps0_iso = 10**(-6.8346 * (T01/T)**1.261 + 4.6151)
    
    ps_ps0 = ps / ps0
    
    # h is the humidity in percent molar concentration
    h_iso = hr * psat_ps0_iso / ps_ps0
    
    # Speed of sound
    c0 = 331  # c0 is the reference sound speed
    c_iso = (1 + 0.16 * h_iso / 100) * c0 * np.sqrt(T / T01)
    
    # ISO formula
    taur = T / T0
    pr = ps / ps0
    
    fr0 = pr * (24 + 40400 * h_iso * (0.02 + h_iso) / (0.391 + h_iso))
    frN = pr * (taur)**(-1/2) * (9 + 280 * h_iso * np.exp(-4.17 * ((taur)**(-1/3) - 1)))
    
    b1 = 0.1068 * np.exp(-3352/T) / (frN + f**2 / frN)
    b2 = 0.01275 * np.exp(-2239.1/T) / (fr0 + f**2 / fr0)
    
    # Calculate the air absorption in dB/meter for the ISO standard
    alpha_iso = -8.686 * f**2 * taur**(1/2) * (1.84e-11 / pr + taur**(-3) * (b1 + b2))
    
    return alpha_iso, c_iso 