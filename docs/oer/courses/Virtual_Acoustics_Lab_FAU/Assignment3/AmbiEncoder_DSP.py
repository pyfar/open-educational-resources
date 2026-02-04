import numpy as np

class AmbiEncoder_DSP:
    """
    Simple ambisonics (virtual-source) encoder
    
    Attributes:
        ambiOrder (int): Ambisonics order used for encoding
        numAmbiCh (int): Number of Ambisonics channels, depends on the order
        doa_azi (float): DOA of virtual source(s) to render, azimuth (rad)
        doa_ele (float): DOA of virtual source(s) to render, elevation (rad)
        numberOfInputs (int): Number of input channels
        numberOfOutputs (int): Number of output channels
    """
    
    def __init__(self, ambiOrder):
        """
        Constructor for AmbiEncoder_DSP
        
        Args:
            ambiOrder (int): Ambisonics order for encoding
        """
        self.ambiOrder = ambiOrder
        self.numAmbiCh = (self.ambiOrder + 1) ** 2
        
        # Initialize DOA
        self.doa_azi = 0
        self.doa_ele = 0
        
        self.numberOfInputs = 1
        self.numberOfOutputs = (self.ambiOrder + 1) ** 2
        
        self.check_config()
    
    def check_config(self):
        """Check if the configuration is valid"""
        assert self.ambiOrder >= 0, "Ambisonics order must be non-negative"
        assert self.numberOfInputs == 1, "Encoder expects single input"
        assert self.numberOfOutputs == self.numAmbiCh, "Output channels must match ambisonics channels"
    
    def set_doa(self, azi, ele):
        """
        Set the direction of arrival
        
        Args:
            azi (np.ndarray): Azimuth angles in radians [Q x 1]
            ele (np.ndarray): Elevation angles in radians [Q x 1]
        """
        # Ensure inputs are column vectors
        assert azi.ndim == 1 or azi.shape[1] == 1, "Azimuth must be a column vector"
        assert ele.ndim == 1 or ele.shape[1] == 1, "Elevation must be a column vector"
        
        self.doa_azi = azi
        self.doa_ele = ele 