import numpy as np
from scipy.io import loadmat
import os

class Config:
    def __init__(self, config_type='default'):
        """
        Initialize configuration with different pre-defined setups.
        
        Args:
            config_type (str): Type of configuration to use. Options are:
                - 'default': Default configuration with concentric rings
                - 'regularLsArray': Regular loudspeaker array using t-design
                - 'fivePointZeroPointFourLsArray': 5.0.4 loudspeaker setup
            tdesign_filename (str, optional): Path to t-design .mat file for AllRAD. If None, use default for config_type.
        """
        self.config_type = config_type
        # Set tdesign_filename before calling config methods
        if config_type == 'regularLsArray':
            self.tdesign_filename = 'tdesign_7.mat'
            self.regular_ls_array_config()
        elif config_type == 'fivePointZeroPointFourLsArray':
            self.tdesign_filename = 'tdesign_5200points.mat'
            self.five_point_zero_point_four_ls_array_config()
        elif config_type == 'default':
            self.tdesign_filename = 'tdesign_7.mat'
            self.default_config()
        else:
            raise ValueError(f"Configuration type '{config_type}' not defined")

    def default_config(self):
        """Default configuration with concentric rings and voice of god"""
        self.fs = 48000
        self.speedOfSound = 343
        self.blockSize = 256
        self.temperature = 20
        self.relativeHumidity = 50
        self.headRadius = 0.09
        self.roomSize = [10, 7, 4]
        self.spatialEncoding = 'object'
        self.ambiOrder = 3
        self.maxre = True
        self.spatialDecoding = 'binaural'
        self.lsPositions = self.default_loudspeakers()
        self.maximumDelay = self.fs
        self.RT60_Low = 2
        self.RT60_High = 0.5
        
        # Early reflection parameters
        self.Early_maxImageSourceOrder = 3
        self.Early_MaximumDelay = self.fs
        
        # Late reflection parameters
        self.Late_numberOfDelays = 8
        self.Late_feedbackMatrix = np.eye(self.Late_numberOfDelays)
        self.Late_MaximumDelay = self.fs

    def default_loudspeakers(self):
        """
        Define a default loudspeaker layout consisting of concentric rings
        and a top speaker (Voice of God). Returns positions in Cartesian coordinates.
        
        Output:
            ls_positions: np.ndarray of shape [num_ls, 3]
        """
        layer_num = [1, 12, 8, 8]
        layer_ele = [90, 0, 45, -45]

        azis = []
        eles = []

        for num, ele in zip(layer_num, layer_ele):
            azi = np.linspace(0, 360 - 360 / num, num)
            el = np.full_like(azi, ele)
            azis.append(azi)
            eles.append(el)

        ls_azi = np.concatenate(azis)
        ls_ele = np.concatenate(eles)

        # Convert degrees to radians
        ls_azi_rad = np.deg2rad(ls_azi)
        ls_ele_rad = np.deg2rad(ls_ele)

        # Convert spherical to Cartesian
        ls_x = np.cos(ls_ele_rad) * np.cos(ls_azi_rad)
        ls_y = np.cos(ls_ele_rad) * np.sin(ls_azi_rad)
        ls_z = np.sin(ls_ele_rad)

        ls_positions = np.stack((ls_x, ls_y, ls_z), axis=-1)

        return ls_positions

    def regular_ls_array_config(self):
        """Configuration using t-design for regular loudspeaker array"""
        self.default_config()
        # Load t-design points from MATLAB file
        tdesign_data = loadmat(self.tdesign_filename)
        self.lsPositions = tdesign_data['tdesign_7']

    def five_point_zero_point_four_ls_array_config(self):
        """Configuration for 5.0.4 loudspeaker setup"""
        self.default_config()
        
        # Define loudspeaker positions in spherical coordinates
        ls_azi = np.array([0, 30, -30, -110, 110, -60, 60, -135, 135]) * np.pi / 180
        ls_ele = np.array([0, 0, 0, 0, 0, 45, 45, 45, 45]) * np.pi / 180
        
        # Convert to Cartesian coordinates
        ls_x = np.cos(ls_ele) * np.cos(ls_azi)
        ls_y = np.cos(ls_ele) * np.sin(ls_azi)
        ls_z = np.sin(ls_ele)
        
        self.lsPositions = np.stack((ls_x, ls_y, ls_z), axis=-1) 

    @property
    def numLoudspeakers(self):
        return self.lsPositions.shape[0]

    @property
    def loudspeakerPositions(self):
        # Return lsPositions in degrees for compatibility if needed
        # If lsPositions is already in radians, convert to degrees
        # Otherwise, just return lsPositions
        # Here, we assume lsPositions is in cartesian, so return as is
        return self.lsPositions

    def __repr__(self):
        return f"Config(config_type='{self.config_type}')" 