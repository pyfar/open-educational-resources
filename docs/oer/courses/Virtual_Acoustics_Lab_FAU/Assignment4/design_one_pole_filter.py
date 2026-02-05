import numpy as np

def design_one_pole_filter(HDc, HNyq):
    """
    Compute one pole filter.
    
    Args:
        HDc (np.ndarray): Linear magnitude at DC of size [1, number of filters]
        HNyq (np.ndarray): Linear magnitude at Nyquist of size [1, number of filters]
        
    Returns:
        np.ndarray: SOS filters of size [6 x number of filters]
        
    Example:
        sos = design_one_pole_filter(np.array([1, 0.9]), np.array([0.7, 0.5]))
    """
    num_filters = len(HDc)
    sos = np.zeros((6, num_filters))
    sos[3, :] = 1  # a0 = 1
    
    # TASK: Implement one-pole filter
    # a) See: https://ccrma.stanford.edu/~jos/fp/One_Pole.html 
    # b) Solve for a1 and b0 (e.g. by plugging in HDc and HNyq and rearranging)
    # c) Implement the filter coefficients into second-order sections (SOSs) 
    # [b0; b1; b2; a0; a1; a2], where a0 = 1.
    # d) Vectorize for multiple filters
    
    # SOLUTION
    r = HDc / HNyq
    
    a1 = (1 - r) / (1 + r)
    b0 = (1 - a1) * HNyq
    
    sos[0, :] = b0  # b0
    sos[4, :] = a1  # a1
    
    return sos 